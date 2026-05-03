import asyncio
import logging
import time
import typing as tp
import torch
import numpy as np
import random
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger('async_pipeline_engine')


# ------------------------------------------------------------
# GreenContext / Execution Context (EC) Data Structures
# ------------------------------------------------------------

@dataclass
class WQConfig:
    """Work Queue configuration for an Execution Context."""
    device: int = 0
    sharing_scope: str = "GreenCtxBalanced"  # "GreenCtxBalanced" | "DeviceCtxBalanced"
    wq_concurrency_limit: int = 8  # max concurrent streams sharing this WQ


@dataclass
class SMResource:
    """SM (Streaming Multiprocessor) resource allocation for an EC."""
    sm_count: int = 0  # 0 = discover mode (API fills in actual value)
    min_sm_partition_size: int = 0
    sm_coscheduled_alignment: int = 0
    coscheduled_sm_count: int = 0  # Thread Block Cluster size; 0 = device default
    preferred_coscheduled_sm_count: int = 0  # CC10.0+ preferred cluster dims
    flags: int = 0  # 0 or cudaDevSmResourceGroupBackfill


@dataclass
class GreenContextConfig:
    """
    Describes the resources bound to a single Execution Context (EC).
    Mirrors the cudaDevResourceDesc / cuDevResourceDesc abstraction.
    """
    model_index: int
    sm_resource: SMResource = field(default_factory=SMResource)
    wq_config: WQConfig = field(default_factory=WQConfig)
    # sm_allocation_ratio: fraction of total SMs allocated to this EC (0.0-1.0)
    sm_allocation_ratio: float = 1.0
    # ec_handle: opaque handle returned by cudaGreenCtxCreate / cuGreenCtxCreate
    ec_handle: tp.Any = None
    # ec_stream: cudaExecutionCtxStreamCreate / cuGreenCtxStreamCreate handle
    ec_stream: tp.Any = None

    @property
    def sm_count_actual(self) -> int:
        """Returns actual SM count (0 means discover mode, filled by API)."""
        return self.sm_resource.sm_count

    def __repr__(self):
        return (f"GreenContextConfig(model_idx={self.model_index}, "
                f"sm_ratio={self.sm_allocation_ratio:.2f}, "
                f"wq_concurrency={self.wq_config.wq_concurrency_limit})")


# Fallback generation ratios (used when predestined_gen_len is not provided)
ST_ratio = 0.25   # STT: output text tokens ≈ input frames * ST_ratio
TS_ratio = 4      # TTS: output audio frames ≈ input text tokens * TS_ratio
max_seq_len = 50  # LM fallback: max generation length
EOS_prob = 0.05   # LM fallback: EOS probability (geometric distribution)


class RequestStatus(Enum):
    PREFILL = "Prefill"
    DECODE = "Decode"


@dataclass
class Batch:
    req_ids: list[str] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    query_lens: list[int] = field(default_factory=list)
    block_idss: list[list[int]] = field(default_factory=list)
    gen_length: list[int] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.req_ids)

    @property
    def show(self):
        return (f"Batch size: {self.batch_size}, Req IDs: {self.req_ids}, "
                f"Context Lens: {self.context_lens}, Query Lens: {self.query_lens}")


@dataclass
class PipelineRequest:
    req_id: str
    time_stamp: float
    original_input_data: tp.Any
    final_output_queue: asyncio.Queue
    TTNF: list[tuple[float, int]] = field(default_factory=list)

    frame_time_stamp: tp.Optional[float] = None
    out_time_stamp: tp.Optional[float] = None
    current_model_input_tensor: tp.Optional[torch.Tensor] = None
    current_model_idx: int = 0
    generated_tokens_count: int = 0
    target_generation_length: int = 0
    status: RequestStatus = RequestStatus.PREFILL
    original_input_len: int = 0
    predestined_gen_len: list[int] = None


@dataclass
class PipelineIntermediateResult:
    next_model_idx: int
    requests: list[PipelineRequest]
    batch_metadata: Batch
    model_output_tensor: tp.Optional[torch.Tensor] = None
    status: RequestStatus = RequestStatus.PREFILL
    target_timestamp: float = field(default_factory=lambda: float('inf'))

    def show(self):
        logger.info(
            f"PipelineIntermediateResult for model {self.next_model_idx}, "
            f"batch size: {self.batch_metadata.batch_size}, "
            f"output tensor: {self.model_output_tensor.shape if self.model_output_tensor is not None else 'N/A'}"
        )


class BaseModel:
    def __init__(self,
                 model_name: str,
                 model_index: int,
                 model_type: str,
                 device: str = 'cuda',
                 ec_config: tp.Optional[GreenContextConfig] = None,
                 trace_mode: bool = False):
        self.model_name = model_name
        self.model_index = model_index
        self.device = torch.device(device)
        self.model_type = model_type

        # Execution time tracking
        self.total_processing_time = 0.0  # ms
        self.batch_count = 0

        # Throughput tracking
        self.total_requests_processed = 0  # 累计流出的request数量（frame量）
        self.total_rounds_completed = 0    # 累计完成该模型处理的request数量（对话轮次）

        # Steady-state throughput tracking
        self.steady_state_start_time: float = None  # 稳态起始时间
        self.ss_total_requests_processed = 0  # 稳态后流出的request数量
        self.ss_total_rounds_completed = 0    # 稳态后完成的request数量

        # GreenContext / EC resources
        self.ec_config = ec_config
        # SM shortage factor: fewer SMs = proportionally slower
        # If sm_allocation_ratio = 0.5, processing is 2x slower (1/0.5 = 2.0)
        self.sm_shortage_factor = (
            1.0 / ec_config.sm_allocation_ratio
            if ec_config is not None and ec_config.sm_allocation_ratio > 0
            else 1.0
        )
        # WQ concurrency affects how efficiently batches can be formed
        self.wq_concurrency_limit = (
            ec_config.wq_config.wq_concurrency_limit
            if ec_config is not None
            else 8
        )

        # Trace recording
        self.trace_mode = trace_mode
        self.trace_batches: list[dict] = []

        logger.info(f"Initialized BaseModel '{self.model_name}_{self.model_index}' on {self.device} "
                    f"(SM factor={self.sm_shortage_factor:.2f}, WQ limit={self.wq_concurrency_limit}, "
                    f"trace={trace_mode})")

        with open("./mini_vllm/perf_data.json", "r") as f:
            config_data = json.load(f)

        if model_name in config_data:
            self.perf_config = config_data[model_name]
        else:
            self.perf_config = config_data.get("default", {})
            logger.warning(f"No performance config for '{model_name}', using default.")

        self.prefill_config = self.perf_config.get("prefill")
        self.decode_config = self.perf_config.get("decode")

    def get_trace(self) -> list[dict]:
        """Return recorded trace batches."""
        return self.trace_batches

    def reset_trace(self):
        """Clear recorded trace."""
        self.trace_batches = []

    def set_steady_state_start(self, start_time: float):
        self.steady_state_start_time = start_time

    def get_time_estimate(self, batch_metadata: Batch) -> float:
        B = batch_metadata.batch_size
        if B == 0:
            return 0.001

        Q_max = max(batch_metadata.query_lens) if batch_metadata.query_lens else 1
        C_max = max(batch_metadata.context_lens) if batch_metadata.context_lens else 1

        prefill_time = (self.prefill_config[0] * (B * Q_max ** 2)
                        + self.prefill_config[1] * (B * Q_max)
                        + self.prefill_config[2]) if self.prefill_config else 0.001
        decode_time = (self.decode_config[0] * (B * C_max)
                       + self.decode_config[1] * B
                       + self.decode_config[2]) if self.decode_config else 0.001

        # Scale by SM shortage factor (fewer SMs = proportionally higher latency)
        prefill_time *= self.sm_shortage_factor
        decode_time *= self.sm_shortage_factor

        if batch_metadata.query_lens and batch_metadata.query_lens[0] > 1:
            return max(0.001, prefill_time)
        else:
            if decode_time > 5:
                with open("long_decode_batches.log", "a") as f:
                    f.write(f"Long decode time: {decode_time:.4f}ms for batch size {B}, context {C_max}, "
                            f"SM factor {self.sm_shortage_factor:.2f}\n")
            return max(0.001, decode_time)

    async def process_batch(self, current_batch_result: PipelineIntermediateResult) -> PipelineIntermediateResult:
        # Capture trace at batch level BEFORE processing
        if self.trace_mode and current_batch_result.requests:
            batch_meta = current_batch_result.batch_metadata
            status = current_batch_result.status
            # Calculate frame counts
            if status == RequestStatus.PREFILL:
                frames_in = sum(batch_meta.query_lens)
                frames_out = 0
            else:
                frames_in = sum(batch_meta.context_lens)
                frames_out = len(current_batch_result.requests)  # 1 token per request per decode step
            self.trace_batches.append({
                'batch_size': batch_meta.batch_size,
                'req_ids': list(batch_meta.req_ids),
                'query_lens': list(batch_meta.query_lens),
                'context_lens': list(batch_meta.context_lens),
                'status': status.value,
                'timestamp': time.perf_counter(),
                'frames_in': frames_in,
                'frames_out': frames_out,
            })

        if not current_batch_result.requests:
            return PipelineIntermediateResult(
                next_model_idx=current_batch_result.next_model_idx,
                requests=[],
                batch_metadata=Batch(),
                model_output_tensor=None,
                status=current_batch_result.status,
            )

        batch_metadata = current_batch_result.batch_metadata
        logger.debug(f"Model '{self.model_name}_{self.model_index}' received batch: {batch_metadata.show}")

        B = batch_metadata.batch_size
        if B == 0:
            return PipelineIntermediateResult(
                next_model_idx=current_batch_result.next_model_idx,
                requests=current_batch_result.requests,
                batch_metadata=batch_metadata,
                model_output_tensor=None,
                status=current_batch_result.status,
            )

        Q_max = max(batch_metadata.query_lens) if batch_metadata.query_lens else 1
        C_max = max(batch_metadata.context_lens) if batch_metadata.context_lens else 1

        prefill_time = (self.prefill_config[0] * (B * Q_max ** 2)
                        + self.prefill_config[1] * (B * Q_max)
                        + self.prefill_config[2]) if self.prefill_config else 0.001
        decode_time = (self.decode_config[0] * (B * C_max)
                       + self.decode_config[1] * B
                       + self.decode_config[2]) if self.decode_config else 0.001

        # Scale by SM shortage factor (GreenContext resource partitioning)
        prefill_time *= self.sm_shortage_factor
        decode_time *= self.sm_shortage_factor

        if current_batch_result.status == RequestStatus.PREFILL:
            # Prefill: process all input tokens; one decode step follows per generated token
            total_time = max(0.001, prefill_time)
            simulated_output_tokens_per_request = Q_max
        else:
            # Decode: one token generated per request per step
            total_time = max(0.001, decode_time)
            simulated_output_tokens_per_request = 1

        time.sleep(total_time / 1000)  # total_time is in ms

        batch_end_time = time.perf_counter()

        # Track execution time
        self.total_processing_time += total_time
        self.batch_count += 1

        # Throughput: each request in batch is processed (counts as frame)
        self.total_requests_processed += B
        if self.steady_state_start_time is not None and batch_end_time >= self.steady_state_start_time:
            self.ss_total_requests_processed += B

        new_requests_metadata = []
        model_output_tensor_list = []

        for i, req in enumerate(current_batch_result.requests):
            req.generated_tokens_count += simulated_output_tokens_per_request

            output_tensor_for_req = torch.randn(simulated_output_tokens_per_request, device=self.device)
            model_output_tensor_list.append(output_tensor_for_req)

            req.TTNF.append((time.perf_counter(), req.current_model_idx))

            if req.status == RequestStatus.PREFILL:
                req.status = RequestStatus.DECODE
                # Set decode target: use predestined length if available, else fall back to model-type heuristic
                if req.predestined_gen_len:
                    req.target_generation_length = req.predestined_gen_len[0]
                    req.predestined_gen_len = req.predestined_gen_len[1:]
                else:
                    if self.model_type == "STT":
                        req.target_generation_length = int(req.original_input_len * ST_ratio)
                    elif self.model_type == "TTS":
                        req.target_generation_length = int(req.target_generation_length * TS_ratio)
                    else:  # LM
                        req.target_generation_length = min(5 + np.random.geometric(EOS_prob), max_seq_len)

                req.generated_tokens_count = 0
                # KV-cache after prefill contains the original input context
                req.current_model_input_tensor = torch.randn(req.original_input_len, device=self.device)

            elif req.status == RequestStatus.DECODE:
                # Extend KV-cache by appending the newly generated token
                try:
                    req.current_model_input_tensor = (
                        torch.cat((req.current_model_input_tensor, output_tensor_for_req), dim=-1)
                        if req.current_model_input_tensor is not None
                        else output_tensor_for_req.unsqueeze(0)
                    )
                except Exception as e:
                    logger.error(f"Tensor concat error for request {req.req_id}: {e}")
                    raise

            # Advance to next model once this stage's generation target is reached
            if req.status == RequestStatus.DECODE and req.generated_tokens_count >= req.target_generation_length:
                req.current_model_idx += 1
                req.status = RequestStatus.PREFILL
                req.generated_tokens_count = 0
                self.total_rounds_completed += 1  # Request completed this model's round
                if self.steady_state_start_time is not None and batch_end_time >= self.steady_state_start_time:
                    self.ss_total_rounds_completed += 1

            new_requests_metadata.append(req)

        model_output_tensor = (
            torch.cat([t.unsqueeze(0) for t in model_output_tensor_list], dim=0)
            if model_output_tensor_list else None
        )

        return PipelineIntermediateResult(
            next_model_idx=self.model_index,
            requests=new_requests_metadata,
            batch_metadata=Batch(),
            model_output_tensor=model_output_tensor,
            status=current_batch_result.status,
        )


class AsyncPipelineEngine:
    def __init__(self,
                 model_factories: tp.Sequence[tp.Callable[[], BaseModel]],
                 device: str = 'cuda',
                 max_active_batch_size: int = 4,
                 batch_timeout: float = 0.01,
                 time_delay_prefill: float = 2.0,
                 time_delay_decode: float = 0.02,
                 ec_configs: tp.Optional[list[tp.Optional[GreenContextConfig]]] = None,
                 trace_model_name: tp.Optional[str] = None):
        """
        Args:
            model_factories: Ordered list of callables that each construct one pipeline stage.
            max_active_batch_size: Trigger batching once this many requests are queued for a model.
            batch_timeout: Also trigger batching if this many seconds have passed since the last batch.
            time_delay_prefill: SLO deadline increment (s) after each prefill step.
            time_delay_decode: SLO deadline increment (s) after each decode step.
            ec_configs: Optional list of GreenContextConfig (one per model) for resource partitioning.
            trace_model_name: If set, enable trace mode for the model with this model_type (e.g. "LM").
        """
        self.device = device
        self.max_active_batch_size = max_active_batch_size
        self.batch_timeout = batch_timeout
        self.time_delay_prefill = time_delay_prefill
        self.time_delay_decode = time_delay_decode
        self.ec_configs = ec_configs
        self.trace_model_name = trace_model_name

        self.main_queue: asyncio.PriorityQueue[tuple[float, PipelineIntermediateResult]] = asyncio.PriorityQueue()

        # Pass ec_config to each model factory if provided
        self.model_pipeline: list[BaseModel] = []
        for idx, factory in enumerate(model_factories):
            ec_cfg = ec_configs[idx] if ec_configs and idx < len(ec_configs) else None
            # Determine if this model should be traced
            trace_this = (trace_model_name is not None)
            # Factory signature may or may not accept ec_config; try both ways
            try:
                model = factory(ec_config=ec_cfg)
            except TypeError:
                model = factory()
                if ec_cfg is not None:
                    # Retroactively apply EC config if factory doesn't accept it
                    model.ec_config = ec_cfg
                    model.sm_shortage_factor = (
                        1.0 / ec_cfg.sm_allocation_ratio
                        if ec_cfg.sm_allocation_ratio > 0 else 1.0
                    )
                    model.wq_concurrency_limit = ec_cfg.wq_config.wq_concurrency_limit
            # Apply trace mode to matching model
            if trace_this and model.model_type == trace_model_name:
                model.trace_mode = True
                model.trace_batches = []
                logger.info(f"  Trace mode ENABLED for {model.model_type}")
            self.model_pipeline.append(model)

        if not self.model_pipeline:
            raise ValueError("Model pipeline cannot be empty.")

        # Per-model queues: pending_requests holds newly arrived requests awaiting admission;
        # active_requests_per_model holds admitted requests ready to be batched.
        self.pending_requests: list[PipelineRequest] = []
        self.active_requests_per_model: list[list[PipelineRequest]] = [[] for _ in self.model_pipeline]

        self.batching_data: list[int] = []
        self.last_batch_time = [time.perf_counter()] * len(self.model_pipeline)

        # Telemetry counters (sampled every sampling_interval seconds)
        self.queuing_task_counter = [[0] for _ in range(len(self.model_pipeline))]
        self.unit_queuing_task_counter = [[0] for _ in range(len(self.model_pipeline))]
        self.batch_counter = [[0] for _ in range(len(self.model_pipeline))]
        self.unit_batch_counter = [[0] for _ in range(len(self.model_pipeline))]
        self.last_count_time = time.perf_counter()
        self.sampling_interval = 1.0

        self.queue_status_observer = [0] * len(self.model_pipeline)

        logger.info(f"AsyncPipelineEngine initialized with {len(self.model_pipeline)} models.")
        logger.info(f"Max active batch size: {max_active_batch_size}, timeout: {batch_timeout}s")
        if ec_configs:
            for cfg in ec_configs:
                if cfg:
                    logger.info(f"  EC config: {cfg}")

    async def add_request(self,
                          req_id: str,
                          input_data: tp.Any,
                          target_model_idx: int = 0,
                          predestined_gen_length: list[int] = None) -> PipelineRequest:
        output_queue = asyncio.Queue()
        initial_input_len = len(str(input_data))
        request = PipelineRequest(
            req_id=req_id,
            time_stamp=time.perf_counter(),
            original_input_data=input_data,
            current_model_input_tensor=torch.randn(initial_input_len, device=self.device),
            final_output_queue=output_queue,
            frame_time_stamp=time.perf_counter() + self.time_delay_prefill,
            current_model_idx=target_model_idx,
            original_input_len=initial_input_len,
            status=RequestStatus.PREFILL,
            predestined_gen_len=predestined_gen_length,
        )
        self.pending_requests.append(request)
        logger.debug(f"Request {req_id} added to pending, targeting model {target_model_idx}.")
        return request

    async def _continuous_batching_scheduler(self):
        """
        Admits pending requests into per-model active queues, then constructs and
        dispatches batches (prefill and decode separately) for each model stage.

        WQ Concurrency Awareness: each model's ec_config carries a wq_concurrency_limit.
        When the number of concurrent batches (in-flight requests) exceeds this limit,
        WQ false-dependencies serialize batches, degrading effective throughput.
        We model this by capping the effective batch size and logging warnings.
        """
        while True:
            # Admit all pending requests to their target model's active queue
            while self.pending_requests:
                req = self.pending_requests.pop(0)
                self.active_requests_per_model[req.current_model_idx].append(req)
                logger.debug(f"Request {req.req_id} admitted to model {req.current_model_idx} queue.")

            for model_idx, active_reqs in enumerate(self.active_requests_per_model):
                timeout_elapsed = (
                    time.perf_counter() - self.last_batch_time[model_idx] >= self.batch_timeout
                )
                if not active_reqs or (not timeout_elapsed and len(active_reqs) < self.max_active_batch_size):
                    continue

                # WQ concurrency cap: effective batch size is limited by wq_concurrency_limit
                # When active requests exceed this, WQ introduces serialization overhead
                model = self.model_pipeline[model_idx]
                wq_limit = getattr(model, 'wq_concurrency_limit', 8)
                wq_saturation = len(active_reqs) / wq_limit if wq_limit > 0 else 1.0

                # Effective max batch size considering WQ saturation
                # High saturation (>1.0) means WQ is overloaded; reduce batch size to simulate slowdown
                effective_max_batch = min(
                    self.max_active_batch_size,
                    max(1, int(self.max_active_batch_size / max(wq_saturation, 1.0)))
                )

                if wq_saturation > 1.0:
                    logger.debug(
                        f"Model {model_idx} WQ saturation={wq_saturation:.2f} "
                        f"(active={len(active_reqs)}, limit={wq_limit}); "
                        f"effective batch cap={effective_max_batch}"
                    )

                # Separate prefill and decode, sorted by earliest deadline first
                prefill_reqs = sorted(
                    [r for r in active_reqs if r.status == RequestStatus.PREFILL],
                    key=lambda r: r.frame_time_stamp,
                )
                decode_reqs = sorted(
                    [r for r in active_reqs if r.status == RequestStatus.DECODE],
                    key=lambda r: r.frame_time_stamp,
                )

                # Dispatch each non-empty group as its own batch, respecting effective cap
                for batch in filter(None, [prefill_reqs, decode_reqs]):
                    # Split oversized batches at WQ concurrency boundary
                    for start in range(0, len(batch), effective_max_batch):
                        sub_batch = batch[start:start + effective_max_batch]
                        for req in sub_batch:
                            if req in active_reqs:
                                active_reqs.remove(req)
                        await self._send_batch(sub_batch, model_idx)

            await asyncio.sleep(0.001)

    async def _send_batch(self, batch: list[PipelineRequest], model_idx: int):
        req_ids_batch, input_ids_batch, context_lens_batch, query_lens_batch, block_idss_batch = [], [], [], [], []
        batch_status = (
            RequestStatus.PREFILL if any(r.status == RequestStatus.PREFILL for r in batch)
            else RequestStatus.DECODE
        )

        self.batching_data.append(len(batch))

        for req in batch:
            req_ids_batch.append(req.req_id)
            block_size = req.current_model_input_tensor.shape[-1] // 4 if req.current_model_input_tensor is not None else 1
            block_idss_batch.append(list(range(block_size)))

            if req.status == RequestStatus.PREFILL:
                # Prefill: query covers the full input; no prior KV-cache context
                context_lens_batch.append(0)
                query_lens_batch.append(req.original_input_len)
                input_ids_batch.append(req.original_input_len)
            else:
                # Decode: generate one token; KV-cache context = original prompt + generated so far
                context_lens_batch.append(req.original_input_len + req.generated_tokens_count)
                query_lens_batch.append(1)
                input_ids_batch.append(1)

        batch_metadata = Batch(
            req_ids=req_ids_batch,
            input_ids=input_ids_batch,
            context_lens=context_lens_batch,
            query_lens=query_lens_batch,
            block_idss=block_idss_batch,
        )

        intermediate_result = PipelineIntermediateResult(
            next_model_idx=model_idx,
            requests=batch,
            batch_metadata=batch_metadata,
            model_output_tensor=None,
            status=batch_status,
            target_timestamp=min(req.frame_time_stamp for req in batch),
        )

        for req in batch:
            assert req.current_model_idx == model_idx, (
                f"Request {req.req_id} model index mismatch: expected {model_idx}, got {req.current_model_idx}"
            )

        await self.main_queue.put((intermediate_result.target_timestamp, intermediate_result))
        self.queue_status_observer[model_idx] += 1
        self.last_batch_time[model_idx] = time.perf_counter()
        logger.debug(f"Batch queued for model {model_idx}: {len(batch)} requests ({batch_status.value}).")

    async def run(self):
        logger.info('Starting AsyncPipelineEngine (Continuous Batching)...')
        scheduler_task = asyncio.create_task(self._continuous_batching_scheduler())

        try:
            while True:
                if self.main_queue.empty():
                    await asyncio.sleep(0.001)
                    continue

                if time.perf_counter() - self.last_count_time >= self.sampling_interval:
                    for i in range(len(self.model_pipeline)):
                        self.queuing_task_counter[i].append(np.mean(self.unit_queuing_task_counter[i]))
                        self.batch_counter[i].append(np.mean(self.unit_batch_counter[i]))
                    self.unit_queuing_task_counter = [[0] for _ in range(len(self.model_pipeline))]
                    self.unit_batch_counter = [[0] for _ in range(len(self.model_pipeline))]
                    self.last_count_time = time.perf_counter()

                _, current_result = await self.main_queue.get()

                if not current_result.requests:
                    self.main_queue.task_done()
                    continue

                model_indices = [req.current_model_idx for req in current_result.requests]
                if len(set(model_indices)) != 1:
                    raise ValueError(f"Inconsistent model indices in batch: {model_indices}")
                target_model_idx = model_indices[0]

                self.queue_status_observer[target_model_idx] -= 1
                for i in range(len(self.model_pipeline)):
                    self.unit_queuing_task_counter[i].append(self.queue_status_observer[i])
                self.unit_batch_counter[target_model_idx].append(current_result.batch_metadata.batch_size)

                processed_result = await self.model_pipeline[target_model_idx].process_batch(current_result)

                # Remove all processed requests from the model's active queue (they'll be re-admitted below)
                processed_req_ids = {req.req_id for req in processed_result.requests}
                self.active_requests_per_model[target_model_idx] = [
                    r for r in self.active_requests_per_model[target_model_idx]
                    if r.req_id not in processed_req_ids
                ]

                for req in processed_result.requests:
                    if req.current_model_idx >= len(self.model_pipeline):
                        # Request has completed all pipeline stages
                        if processed_result.model_output_tensor is not None:
                            final_output_data = req.current_model_input_tensor.cpu().numpy()
                            await req.final_output_queue.put((time.perf_counter(), final_output_data))
                            req.out_time_stamp = time.perf_counter()
                            logger.debug(f"Request {req.req_id} completed pipeline.")
                        else:
                            logger.warning(f"Request {req.req_id} finished pipeline but produced no output.")
                    else:
                        # Advance deadline and re-admit to the appropriate model queue
                        if req.status == RequestStatus.DECODE:
                            req.frame_time_stamp += self.time_delay_decode
                        else:
                            req.frame_time_stamp += self.time_delay_prefill
                        self.active_requests_per_model[req.current_model_idx].append(req)

                self.main_queue.task_done()
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in AsyncPipelineEngine runner: {e}")
        finally:
            scheduler_task.cancel()
            try:
                await scheduler_task
            except Exception:
                pass
            logger.info('AsyncPipelineEngine runner finished.')


async def main():
    logging.basicConfig(level=logging.INFO)

    lambda_rate = 4
    num_requests = 20
    batch_size = 4

    def create_stt():
        return BaseModel(model_name="1.3B", model_index=0, model_type="STT", device='cpu')
    def create_lm():
        return BaseModel(model_name="1.3B", model_index=1, model_type="LM", device='cpu')
    def create_tts():
        return BaseModel(model_name="1.3B", model_index=2, model_type="TTS", device='cpu')

    engine = AsyncPipelineEngine(
        model_factories=[create_stt, create_lm, create_tts],
        device='cpu',
        max_active_batch_size=batch_size,
    )
    engine_task = asyncio.create_task(engine.run())

    requests = []
    for i in range(num_requests):
        await asyncio.sleep(random.expovariate(lambda_rate))
        request = await engine.add_request(f"req-{i}", f"This is request {i} data.")
        requests.append(request)

    latencies, TTFFs, TBFs = [], [], []
    for req in requests:
        timestamp, result = await req.final_output_queue.get()
        latency = (req.out_time_stamp or timestamp) - req.time_stamp
        latencies.append(latency)

        ttnf_times = [t for t, _ in req.TTNF]
        ttff = (ttnf_times[0] - req.time_stamp) if ttnf_times else 0
        TTFFs.append(ttff)
        TBFs.extend(ttnf_times[j] - ttnf_times[j - 1] for j in range(1, len(ttnf_times)))

        logger.info(f"Request {req.req_id}: latency={latency:.4f}s, TTFF={ttff:.4f}s")

    avg_tbf = 1000 * np.mean(TBFs) if TBFs else float('nan')
    logger.info(
        f"Avg latency: {1000*np.mean(latencies):.2f}ms, "
        f"Avg TTFF: {1000*np.mean(TTFFs):.2f}ms, "
        f"Avg TBF: {avg_tbf:.2f}ms"
    )

    with open("run_log.txt", "a") as f:
        f.write(f"\nNum requests: {num_requests}, Lambda: {lambda_rate}, Batch: {batch_size}\n")
        f.write(f"Avg latency: {1000*np.mean(latencies):.2f}ms, Avg TTFF: {1000*np.mean(TTFFs):.2f}ms\n")

    await asyncio.sleep(5)
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        logger.info("Engine task cancelled.")


logger.info("\n--- All results received and engine shut down ---")


# ============================================================
# Trace & Replay: Measure single-model throughput at 100% SM
# ============================================================

def replay_model_throughput(
    model_name: str,
    model_type: str,
    trace_batches: list[dict],
    sm_allocation_ratio: float = 1.0,
) -> dict:
    """
    Replay a recorded trace with a single model at given SM allocation.

    Args:
        model_name: e.g. "1.3B", "13B"
        model_type: e.g. "STT", "LM", "TTS"
        trace_batches: list of traced batch dicts from BaseModel.get_trace()
        sm_allocation_ratio: SM allocation (1.0 = 100%, no slowdown)

    Returns:
        dict with throughput metrics:
        - total_frames_in: sum of input frames
        - total_frames_out: sum of output frames
        - total_time_s: wall-clock time for all batch processing
        - throughput_in: frames/s input
        - throughput_out: frames/s output
        - batch_count: number of batches
        - avg_batch_size: average batch size
    """
    # Create model with 100% SM allocation
    ec_config = GreenContextConfig(
        model_index=0,
        sm_resource=SMResource(),
        wq_config=WQConfig(wq_concurrency_limit=8),
        sm_allocation_ratio=sm_allocation_ratio,
    )
    model = BaseModel(
        model_name=model_name,
        model_index=0,
        model_type=model_type,
        device='cpu',
        ec_config=ec_config,
    )

    # Load perf config
    with open("./mini_vllm/perf_data.json", "r") as f:
        config_data = json.load(f)
    if model_name in config_data:
        perf = config_data[model_name]
    else:
        perf = config_data.get("default", {})

    prefill_cfg = perf.get("prefill", [2e-6, 4e-2, 7.0])
    decode_cfg = perf.get("decode", [7e-5, 4e-1, 0.013])

    # SM shortage factor (use provided ratio, not model internal factor)
    shortage_factor = 1.0 / sm_allocation_ratio if sm_allocation_ratio > 0 else 1.0

    total_frames_in = 0
    total_frames_out = 0
    total_processing_ms = 0.0
    batch_sizes = []
    unique_req_ids = set()
    start_time = time.perf_counter()

    for trace in trace_batches:
        batch_size = trace['batch_size']
        query_lens = trace['query_lens']
        context_lens = trace['context_lens']
        status = trace['status']
        frames_in = trace['frames_in']
        frames_out = trace['frames_out']

        total_frames_in += frames_in
        total_frames_out += frames_out
        batch_sizes.append(batch_size)

        # Count unique requests to estimate rounds completed
        if 'req_ids' in trace:
            unique_req_ids.update(trace['req_ids'])

        # Compute processing time using the same formula as BaseModel
        B = batch_size
        Q_max = max(query_lens) if query_lens else 1
        C_max = max(context_lens) if context_lens else 1

        if status == 'Prefill':
            prefill_time = (prefill_cfg[0] * (B * Q_max ** 2)
                           + prefill_cfg[1] * (B * Q_max)
                           + prefill_cfg[2])
            decode_time = 0
        else:  # Decode
            prefill_time = 0
            decode_time = (decode_cfg[0] * (B * C_max)
                         + decode_cfg[1] * B
                         + decode_cfg[2])

        total_time = max(0.001, prefill_time + decode_time) * shortage_factor
        total_processing_ms += total_time

    elapsed_time = time.perf_counter() - start_time

    # Calculate rounds completed based on unique requests processed
    # Each unique req_id that appears in any batch = one round completed at this model
    # Fall back to counting batch entries if req_ids not available (backwards compat)
    all_req_ids = set()
    for trace in trace_batches:
        if 'req_ids' in trace and trace['req_ids']:
            all_req_ids.update(trace['req_ids'])

    if all_req_ids:
        # Have req_ids - count unique requests
        total_rounds = len(all_req_ids)
    else:
        # No req_ids - sum batch sizes as proxy for unique request-processings
        # Each batch processes some requests, so total batches = rounds processed
        total_rounds = len(trace_batches)

    rounds_per_second = total_rounds / (total_processing_ms / 1000.0) if total_processing_ms > 0 else 0

    return {
        'model_name': model_name,
        'model_type': model_type,
        'sm_allocation': sm_allocation_ratio,
        'total_frames_in': total_frames_in,
        'total_frames_out': total_frames_out,
        'total_time_s': total_processing_ms / 1000.0,
        'total_processing_time_ms': total_processing_ms,
        'throughput_in': total_frames_in / (total_processing_ms / 1000.0) if total_processing_ms > 0 else 0,
        'throughput_out': total_frames_out / (total_processing_ms / 1000.0) if total_processing_ms > 0 else 0,
        'batch_count': len(trace_batches),
        'avg_batch_size': np.mean(batch_sizes) if batch_sizes else 0,
        'total_rounds': int(total_rounds),
        'round_throughput': rounds_per_second,
    }


def compare_trace_replay(
    model_name: str,
    model_type: str,
    trace_batches: list[dict],
) -> dict:
    """
    Compare throughput at 100% SM (replay) vs the original traced SM allocation.
    Returns comparison metrics for both scenarios.
    """
    # Find original SM allocation from trace (if recorded)
    # Since trace doesn't store SM ratio, we need to know it from the original run
    # For now, just compute at 100% SM
    result_100 = replay_model_throughput(model_name, model_type, trace_batches, sm_allocation_ratio=1.0)

    return result_100


if __name__ == "__main__":
    asyncio.run(main())
