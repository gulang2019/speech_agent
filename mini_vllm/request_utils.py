from asyncio.queues import Queue, QueueEmpty
from dataclasses import dataclass, field
import torch
import asyncio
import logging
import numpy as np
import time
import typing as tp
import json

logger = logging.getLogger('async_pipeline_engine')


@dataclass
class Batch:
    req_ids: list[str] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    query_lens: list[int] = field(default_factory=list)
    block_idss: list[list[int]] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.req_ids)

    @property
    def position_ids(self) -> list[int]:
        return sum([list(range(cl, cl + ql)) for cl, ql in zip(self.context_lens, self.query_lens)], start=[])

    @property
    def seq_lens(self) -> list[int]:
        return [cl + ql for cl, ql in zip(self.context_lens, self.query_lens)]

@dataclass
class PipelineIntermediateResult:
    batch: Batch
    output_tensor: torch.Tensor

class BaseModel:
    def __init__(self, model_name:str,device: str = 'cuda'):
        self.model_name = model_name
        self.device = torch.device(device)

        logger.info(f"Initialized BaseModel on {self.device}")

        with open("./mini_vllm/perf_data.json", "r") as f:
            config_data = json.load(f)

        if model_name in config_data:
            self.perf_config = config_data[model_name]
            logger.info(f"Loaded performance config for {model_name}: {self.perf_config}")
        else:
            self.perf_config = config_data.get("default", {})
            logger.warning(f"No performance config found for {model_name}. Using default settings.")
        self.prefill_config = self.perf_config.get("prefill")
        self.decode_config = self.perf_config.get("decode")

    
    async def forward(self, input_data: tp.Union[Batch, PipelineIntermediateResult]) -> PipelineIntermediateResult:
        """
        根据输入数据（可以是Batch或上一个模型的中间结果）进行前向传播。
        
        Args:
            input_data (tp.Union[Batch, PipelineIntermediateResult]): 
                管道中第一个模型的输入是原始的Batch。
                后续模型的输入是上一个模型返回的PipelineIntermediateResult。

        Returns:
            PipelineIntermediateResult: 包含更新后的Batch信息和当前模型输出的Tensor。
        """
        if isinstance(input_data, Batch):
            current_batch = input_data
            prev_model_output = None
            # 从current_batch构建第一个模型的输入tensor
            # ... (这部分逻辑与方案一中prev_model_output为None时的构建逻辑类似) ...
            model_input_tensor = torch.randint(
                0, 1000, (current_batch.batch_size, max(current_batch.seq_lens) if current_batch.seq_lens else 1),
                device=self.device, dtype=torch.long
            )
        else: # isinstance(input_data, PipelineIntermediateResult)
            current_batch = input_data.batch
            prev_model_output = input_data.output_tensor
            # 使用prev_model_output作为当前模型的输入tensor
            model_input_tensor = prev_model_output # 假设直接使用
        
        B = current_batch.batch_size
        if B == 0:
            return PipelineIntermediateResult(batch=current_batch, output_tensor=torch.empty(0, 10, device=self.device))

        Q = max(current_batch.query_lens) if current_batch.query_lens else 1
        C = max(current_batch.context_lens) if current_batch.context_lens else 1

        prefill_time = self.prefill_config[0]*(B * Q ** 2) + self.prefill_config[1]*(B * Q) + self.prefill_config[2]
        decode_time = self.decode_config[0]*(B * C) + self.decode_config[1]*B + self.decode_config[2]
        total_time = max(0.001, prefill_time + decode_time)

        await asyncio.sleep(total_time)

        output_tensor = torch.randn(B, 10, device=self.device) # 模拟输出

        logger.debug(f"Model '{self.model_name}' forward completed for batch size {B}, took {total_time:.4f}s.")
        
        # 假设在这里模型可能会更新Batch中的某些信息，例如 block_idss
        # current_batch.block_idss = ... # 示例更新

        return PipelineIntermediateResult(batch=current_batch, output_tensor=output_tensor)



@dataclass
class PipelineRequest:
    req_id: str
    input_data: tp.Any # 用户传入的原始请求数据
    output_queue: Queue # 用于返回结果给调用者
    status: str = "WAITING" # 请求状态

class AsyncPipelineEngine:
    def __init__(self,
                 model_pipeline: tp.Sequence[BaseModel],
                 device: str = 'cuda',
                 batch_size_threshold: int = 4, # 达到此数量就发送batch
                 batch_timeout_ms: int = 50): # 批处理超时时间（毫秒）
        
        self.model_pipeline = model_pipeline
        self.device = device
        self.batch_size_threshold = batch_size_threshold
        self.batch_timeout_s = batch_timeout_ms / 1000.0

        self._request_queue: Queue[PipelineRequest] = Queue()
        self._waiting_for_batch: list[PipelineRequest] = []
        self._last_batch_time: float = time.perf_counter()
        
        logger.info(f"AsyncPipelineEngine initialized with {len(model_pipeline)} models.")
        logger.info(f"Batch size threshold: {batch_size_threshold}, Timeout: {self.batch_timeout_s * 1000}ms")

    async def add_request(self, req_id: str, input_data: tp.Any) -> Queue:
        """
        添加一个新请求到系统。
        返回一个输出队列，调用者可以从中获取请求结果。
        """
        output_queue = Queue()
        request = PipelineRequest(
            req_id=req_id,
            input_data=input_data,
            output_queue=output_queue
        )
        await self._request_queue.put(request)
        logger.debug(f"Request {req_id} added to queue.")
        return output_queue

    async def _process_batch(self, current_batch_requests: list[PipelineRequest]):
        """
        处理一个批次的请求，通过模型管道进行推理。
        """
        if not current_batch_requests:
            return

        # 1. 构建 Batch 对象
        req_ids = [r.req_id for r in current_batch_requests]
        # ⚠️ 这里需要根据您实际的input_data来构建Batch中的input_ids, context_lens等
        # 示例：假设input_data是简单的文本长度
        input_ids_batch = []
        context_lens_batch = []
        query_lens_batch = []
        block_idss_batch = []

        # 模拟根据input_data构建batch数据
        for req in current_batch_requests:
            # 这里的逻辑需要根据input_data的实际格式来填充
            # 假设 input_data 是一个字典，包含 'text_length'
            # 或者您可以设计一个更通用的方式来提取这些信息
            text_length = len(str(req.input_data)) # 简单示例
            input_ids_batch.append(text_length)
            context_lens_batch.append(text_length // 2)
            query_lens_batch.append(text_length - (text_length // 2))
            block_idss_batch.append([i for i in range(text_length // 4)]) # 示例

        batch = Batch(
            req_ids=req_ids,
            input_ids=input_ids_batch,
            context_lens=context_lens_batch,
            query_lens=query_lens_batch,
            block_idss=block_idss_batch
        )

        logger.info(f"Processing batch of size {batch.batch_size} for requests: {batch.req_ids}")

        # 2. 模拟模型输入 (这里是一个简单的随机张量，实际应根据Batch内容构建)
        # 假设模型输入是一个形状为 [batch_size, max_seq_len] 的张量
        max_seq_len = max(batch.seq_lens) if batch.seq_lens else 1
        pipeline_result: tp.Union[Batch, PipelineIntermediateResult] = batch # 管道的初始输入是Batch

        for i, model in enumerate(self.model_pipeline):
            logger.debug(f"Batch {batch.req_ids} entering model {model.model_name} ({i+1}/{len(self.model_pipeline)})")
            pipeline_result = await model.forward(pipeline_result) # 传递上一个模型的整个结果
            logger.debug(f"Batch {batch.req_ids} exited model {model.model_name} ({i+1}/{len(self.model_pipeline)})")

        # 最终结果是 PipelineIntermediateResult，从中提取 output_tensor
        if isinstance(pipeline_result, PipelineIntermediateResult):
            final_output_tensor = pipeline_result.output_tensor
            # ... (分发结果的逻辑与方案一相同) ...
            if final_output_tensor is not None:
                for i, req in enumerate(current_batch_requests):
                    result_for_req = final_output_tensor[i].cpu().numpy()
                    await req.output_queue.put((time.perf_counter(), result_for_req))
                    req.status = "COMPLETED"
                    logger.debug(f"Result for request {req.req_id} put into output queue.")
            else:
                logger.warning(f"Pipeline finished for batch {batch.req_ids} but final output tensor was None.")
        else:
            logger.warning(f"Pipeline finished for batch {batch.req_ids} but returned unexpected type: {type(pipeline_result)}")

    async def run(self):
        """
        引擎主循环，负责从请求队列中收集请求并触发批处理。
        """
        logger.info('AsyncPipelineEngine starting...')
        while True:
            try:
                # 尝试获取一个请求，非阻塞地等待0.001秒
                request = await asyncio.wait_for(self._request_queue.get(), timeout=0.001)
                self._waiting_for_batch.append(request)
                self._last_batch_time = time.perf_counter() # 每次收到请求都更新时间戳
                logger.debug(f"Request {request.req_id} added to _waiting_for_batch. Current count: {len(self._waiting_for_batch)}")

            except asyncio.TimeoutError:
                # 队列为空，或者在指定时间内没有新请求
                pass

            # 检查是否满足批处理条件：
            # 1. 达到批处理数量阈值
            # 2. 或者，有请求在等待，并且达到了批处理超时时间
            if (len(self._waiting_for_batch) >= self.batch_size_threshold or
                (self._waiting_for_batch and (time.perf_counter() - self._last_batch_time >= self.batch_timeout_s))):

                batch_to_process = self._waiting_for_batch[:] # 复制列表进行处理
                self._waiting_for_batch.clear() # 清空等待队列
                self._last_batch_time = time.perf_counter() # 重置时间戳

                if batch_to_process:
                    # 提交批处理任务到事件循环，但不等待其完成，实现并发
                    asyncio.create_task(self._process_batch(batch_to_process))
                    logger.info(f"Dispatched a batch of {len(batch_to_process)} requests for processing.")

            await asyncio.sleep(0.001) # 短暂休眠以让出CPU，确保事件循环可以处理其他任务

# 示例使用
async def main():
    logging.basicConfig(level=logging.INFO) # 可以改为DEBUG查看详细日志

    # 1. 自定义一个包含多个模型的pipeline
    model1 = BaseModel(device='cuda',model_name="facebook/opt-1.3b") # 示例，可以在GPU上
    model2 = BaseModel(device='cuda',model_name="facebook/opt-1.3b")
    model_pipeline = [model1, model2]

    engine = AsyncPipelineEngine(
        model_pipeline=model_pipeline,
        device='cpu',
        batch_size_threshold=2, # 达到2个请求就发送批次
        batch_timeout_ms=100    # 最多等待100毫秒发送批次
    )

    # 启动引擎的运行协程
    engine_task = asyncio.create_task(engine.run())

    # 2. 添加多个请求
    output_queues = []
    for i in range(5):
        input_data = f"This is request {i} data."
        output_queue = await engine.add_request(f"req-{i}", input_data)
        output_queues.append(output_queue)
        await asyncio.sleep(0.02) # 模拟请求间隔

    # 3. 从输出队列中获取结果
    results = {}
    for i, q in enumerate(output_queues):
        start_wait = time.perf_counter()
        timestamp, result = await q.get()
        results[f"req-{i}"] = (timestamp, result)
        end_wait = time.perf_counter()
        logger.info(f"Received result for req-{i} in {end_wait - start_wait:.4f}s. Result timestamp: {timestamp:.4f}, Data shape: {result.shape}")

    # 停止引擎任务 (在实际应用中，你可能需要一个更优雅的停止机制)
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        logger.info("Engine task cancelled.")

    logger.info("\n--- All results received ---")
    for req_id, (ts, res) in results.items():
        logger.info(f"{req_id}: {res.flatten()[:5]}...") # 打印部分结果

if __name__ == "__main__":
    asyncio.run(main())