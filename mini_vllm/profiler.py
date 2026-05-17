import os
import torch
import math 
import numpy as np
from typing import cast, Any
import logging
import time
import csv
import random

from dataclasses import field, dataclass 

logger = logging.getLogger(__name__)


from itertools import accumulate
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import init_distributed_environment, ensure_model_parallel_initialized
from vllm.config import ModelConfig, SchedulerConfig, CacheConfig, ParallelConfig, VllmConfig

from vllm.model_executor.model_loader import get_model_loader
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.config import get_layers_from_vllm_config
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.worker.gpu.attn_utils import init_kv_cache, init_attn_backend
from vllm.v1.kv_cache_interface import (
    KVCacheConfig
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata)
from vllm.forward_context import set_forward_context 

from vllm.config import set_current_vllm_config

@dataclass 
class Config:
    model_name: str
    max_memory_utilization: float 
    block_size: int = 16


@dataclass 
class Batch:
    req_ids: list[str] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    query_lens: list[int] = field(default_factory=list)
    block_idss: list[list[int]] = field(default_factory=list)
    
    @property
    def position_ids(self) -> list[int]:
        return sum([list(range(cl, cl + ql)) for cl, ql in zip(self.context_lens, self.query_lens)], start = [])
    
    @property
    def seq_lens(self) -> list[int]:
        return [cl + ql for cl, ql in zip(self.context_lens, self.query_lens)]
    
BatchOutput = dict[str, list[int]]

class ModelRunner:
    def __init__(self,
                 vllm_config: VllmConfig):
        
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        init_distributed_environment()
        ensure_model_parallel_initialized(1, 1)
        self.vllm_config = vllm_config
        model_loader = get_model_loader(vllm_config.load_config)
        self.model = model_loader.load_model(
            vllm_config=vllm_config, model_config=vllm_config.model_config
        )
        from vllm.platforms import current_platform

        self.device = current_platform.device_type
        logger.info(f"loaded {vllm_config.model_config.model}, takes {torch.cuda.device_memory_used(device = self.device) / 1e9:.3f} Gb")
        self.block_size = vllm_config.cache_config.block_size
        self._init_kv_cache()
        
    def _compute_available_memory(self, utilization: float):
        props = torch.cuda.get_device_properties(device = self.device)
        total_memory = props.total_memory
        used_memory = torch.cuda.device_memory_used(device = self.device)
        assert total_memory * utilization > used_memory, f"no memory available, {total_memory}, {used_memory}"
        return int(total_memory * utilization - used_memory) 
        
    
    def _init_kv_cache(self):

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec
        available_memory = self._compute_available_memory(self.vllm_config.cache_config.gpu_memory_utilization)
        kv_cache_configs = get_kv_cache_configs(self.vllm_config, [kv_cache_spec], [available_memory])
        self.kv_cache_config: KVCacheConfig = kv_cache_configs[0]
        logger.info(f'Allocate {available_memory / 1e9:.3f}Gb to KV Cache, #Blocks: {self.kv_cache_config.num_blocks}, #tokens {self.kv_cache_config.num_blocks * self.block_size}')
        attn_backends, self.attn_metadata_builders = init_attn_backend(
            self.kv_cache_config, 
            self.vllm_config, 
            self.device
        )
        kv_caches = []
        init_kv_cache(
            kv_caches, 
            forward_context = self.vllm_config.compilation_config.static_forward_context, 
            kv_cache_config = self.kv_cache_config, 
            attn_backends = attn_backends,
            device = self.device
        )
        
    def _build_attention_meta(
        self,
        batch: Batch
    ):
        query_start_loc_lst = [0] + list(accumulate(batch.query_lens))
        query_start_loc_cpu = torch.tensor(query_start_loc_lst, dtype = torch.int32, device = 'cpu')
        query_start_loc_gpu = query_start_loc_cpu.to(self.device)
        _seq_lens_cpu = torch.tensor(batch.seq_lens, dtype = torch.int32, device = 'cpu')
        seq_lens = _seq_lens_cpu.to(self.device)
        max_seq_len = max(batch.seq_lens)
        num_reqs = len(batch.query_lens)
        max_query_len = max(batch.query_lens)
        num_computed_tokens_cpu = torch.tensor(batch.context_lens, dtype = torch.int32, device = 'cpu')
        num_tokens = sum(batch.query_lens)
        max_num_blocks = int(math.ceil(max_seq_len/self.block_size))
        block_tables = np.full(shape = (num_reqs, max_num_blocks), fill_value = -1, dtype = np.int32)
        for i in range(len(batch.block_idss)):
            block_ids = batch.block_idss[i]
            block_tables[i, :len(block_ids)] = block_ids 
        block_tables = torch.from_numpy(block_tables).to(dtype = torch.int32).to(self.device)
        slot_mappings = []
        assert len(batch.context_lens) == len(batch.query_lens) == len(batch.block_idss)
        for seq_len, ctx_len, block_ids in zip(batch.seq_lens, batch.context_lens, batch.block_idss):
            slot_mapping = [
                block_ids[idx // self.block_size] * self.block_size + idx % self.block_size for idx in range(ctx_len, seq_len)
            ]
            slot_mappings.extend(slot_mapping)
        slot_mappings = torch.tensor(slot_mappings, dtype = torch.int64, device = self.device)
        attn_metadata = {}
        kv_cache_groups = self.kv_cache_config.kv_cache_groups
        for i, (kv_cache_spec, attn_metadata_builder) in enumerate(zip(kv_cache_groups, self.attn_metadata_builders)):
            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc_gpu,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                _seq_lens_cpu=_seq_lens_cpu,
                max_seq_len=max_seq_len,
                _num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
                block_table_tensor=block_tables,
                slot_mapping=slot_mappings,
                causal=True,
            )
            
            metadata = attn_metadata_builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in kv_cache_spec.layer_names:
                attn_metadata[layer_name] = metadata
        
        # validate block coverage before building slot mapping
        for seq_len, block_ids in zip(batch.seq_lens, batch.block_idss):
            need_blocks = (seq_len + self.block_size - 1) // self.block_size
            assert len(block_ids) >= need_blocks, (len(block_ids), need_blocks, seq_len)
        return attn_metadata

    @property
    def num_blocks(self):
        return self.kv_cache_config.num_blocks

    def execute_batch(
        self, batch: Batch 
    ) -> BatchOutput:
        #print(f"Batch shapes - input_ids: {len(batch.input_ids)}, query_lens: {batch.query_lens}, seq_lens: {batch.seq_lens}, context_lens: {batch.context_lens}")
        

        start_time = time.time()
        attn_meta = self._build_attention_meta(batch)
        generated_tokens = {}
        with set_forward_context(attn_metadata=attn_meta, 
                                 vllm_config = self.vllm_config):
            hidden_states = self.model.forward(
                input_ids=torch.tensor(batch.input_ids, dtype = torch.int64, device = self.device),
                positions=torch.tensor(batch.position_ids, dtype = torch.int64, device = self.device)
            )
            logits = self.model.compute_logits(hidden_states)
            tokens = torch.argmax(logits, dim = -1).cpu().tolist()
        
        prev = 0
        for idx, req_id in zip(accumulate(batch.query_lens), batch.req_ids):
            generated_tokens[req_id] = tokens[prev:idx]
            prev = idx
        
        end_time = time.time()


        # csv_file = f"batch_execution_log_{self.vllm_config.model_config.model}.csv"
        # file_exists = os.path.isfile(csv_file)

        # with open(csv_file, 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     if not file_exists:
        #         writer.writerow(['timestamp', 'num_requests', 'input_ids_len', 'query_lens', 'seq_lens', 'context_lens', 'block_idss', 'execution_time_s'])
            
        #     writer.writerow([
        #         datetime.now().isoformat(),
        #         len(batch.req_ids),
        #         len(batch.input_ids),
        #         batch.query_lens,
        #         batch.seq_lens,
        #         batch.context_lens,
        #         batch.block_idss,
        #         f"{end_time - start_time:.3f}"
        #     ])
        return generated_tokens

def get_vllm_config(
    config: Config
):
    model_config = ModelConfig(
        model=config.model_name,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=config.block_size,
        gpu_memory_utilization=config.max_memory_utilization,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config

# @dataclass 
# class Batch:
#     req_ids: list[str] = field(default_factory=list)
#     input_ids: list[int] = field(default_factory=list)
#     context_lens: list[int] = field(default_factory=list)
#     query_lens: list[int] = field(default_factory=list)
#     block_idss: list[list[int]] = field(default_factory=list)
    
#     @property
#     def position_ids(self) -> list[int]:
#         return sum([list(range(cl, cl + ql)) for cl, ql in zip(self.context_lens, self.query_lens)], start = [])
    
#     @property
#     def seq_lens(self) -> list[int]:
#         return [cl + ql for cl, ql in zip(self.context_lens, self.query_lens)]

# if __name__ == "__main__":
#     config = Config(
#             model_name='facebook/opt-1.3b',
#             max_memory_utilization=0.8,
#         )
#     vllm_config = get_vllm_config(config)
#     with set_current_vllm_config(vllm_config):
#         model_runner = ModelRunner(vllm_config)


def create_mock_batch(
    batch_size: int, 
    context_len: int, 
    query_len: int, 
    vocab_size: int, 
    block_size: int
) -> Batch:
    """
    创建一个模拟 Batch。
    
    :param context_len: KV Cache 中已有的历史长度（对于 Prefill 通常是 0）
    :param query_len:   本次模型需要处理的 Token 数量（对于 Decode 通常是 1）
    """
    batch = Batch()
    
    # 1. 生成 Request ID
    batch.req_ids = [f"req_{i}_{random.randint(1000,9999)}" for i in range(batch_size)]
    
    # 2. 设定长度
    batch.context_lens = [context_len] * batch_size
    batch.query_lens = [query_len] * batch_size
    
    # 3. 生成 Input IDs
    # 注意：Input IDs 只包含本次需要计算的 Token (即 query_len 部分)
    # 旧的历史 token 不需要再次作为 input_ids 传入
    all_tokens = []
    for _ in range(batch_size):
        tokens = [random.randint(0, vocab_size - 1) for _ in range(query_len)]
        all_tokens.extend(tokens)
    batch.input_ids = all_tokens
    
    # 4. 生成 Block IDs
    # Block 需要覆盖整个序列长度 (Context + Query)
    total_seq_len = context_len + query_len
    num_blocks_per_seq = (total_seq_len + block_size - 1) // block_size
    
    batch.block_idss = []
    for i in range(batch_size):
        # 分配虚拟 block ID
        seq_blocks = [i * 100 + b for b in range(num_blocks_per_seq)]
        batch.block_idss.append(seq_blocks)
        
    return batch

@dataclass
class TestConfig:
    name: str        # 配置名称，用于打印日志
    batch_size: int
    context_len: int # L
    query_len: int   # 1 for Decode, >1 for Prefill/Chunked

def benchmark_model_runner(
    model_runner, 
    configs: list[TestConfig], 
    vocab_size: int = None, 
    block_size: int = None, 
    num_warmup: int = 2, 
    num_iters: int = 5
):
    """
    执行基准测试，输出可用于性能模型拟合的详细数据。
    """
    # 1. 自动获取 vocab_size 和 block_size
    if vocab_size is None or block_size is None:
        try:
            vocab_size = model_runner.vllm_config.model_config.get_vocab_size()
            block_size = model_runner.vllm_config.cache_config.block_size
        except AttributeError:
            vocab_size = vocab_size or 50256
            block_size = block_size or 16
            print("Warning: Using default vocab_size/block_size.")

    # 2. 打印表头 (增加了 Context/Query 列)
    print(f"{'Config Name':<20} | {'BS':<4} | {'Ctx':<6} | {'Qry':<4} | {'Latency (ms)':<12} | {'Tokens/Step':<12} | {'Tokens/s':<12}")
    print("-" * 90)

    results = [] # 存储结果供后续拟合模型使用

    for cfg in configs:
        # 1. 预热
        for _ in range(num_warmup):
            warmup_batch = create_mock_batch(
                cfg.batch_size, cfg.context_len, cfg.query_len, vocab_size, block_size
            )
            try:
                _ = model_runner.execute_batch(warmup_batch)
            except Exception as e:
                print(f"Warmup failed for {cfg.name}: {e}")
                break
        
        torch.cuda.synchronize()
        
        # 2. 正式测试
        total_time = 0.0
        total_tokens_step = 0 # 每一步处理的 token 总数 (sum of query_lens)
        
        for _ in range(num_iters):
            batch = create_mock_batch(
                cfg.batch_size, cfg.context_len, cfg.query_len, vocab_size, block_size
            )
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = model_runner.execute_batch(batch)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            total_time += (end_time - start_time)
            total_tokens_step += sum(batch.query_lens)
        
        # 3. 计算统计数据
        avg_time_ms = (total_time / num_iters) * 1000
        avg_tokens_per_step = total_tokens_step / num_iters
        throughput_tokens = avg_tokens_per_step / (total_time / num_iters)
        
        print(f"{cfg.name:<20} | {cfg.batch_size:<4} | {cfg.context_len:<6} | {cfg.query_len:<4} | {avg_time_ms:<12.2f} | {avg_tokens_per_step:<12.0f} | {throughput_tokens:<12.0f}")

        # 保存结果用于性能模型拟合
        results.append({
            "batch_size": cfg.batch_size,
            "context_len": cfg.context_len,
            "query_len": cfg.query_len,
            "latency_ms": avg_time_ms,
            "throughput": throughput_tokens
        })
    
    return results

if __name__ == "__main__":
    # 初始化你的环境
    # from mini_vllm import Config, get_vllm_config, ModelRunner
    
    # 这里的初始化代码保持你原本的逻辑
    # config = Config(...)
    # vllm_config = get_vllm_config(config)
    # with set_current_vllm_config(vllm_config):
    #     model_runner = ModelRunner(vllm_config)
    

    config = Config(
            model_name='facebook/opt-1.3b',
            max_memory_utilization=0.8,
            block_size = 16,
        )
    vllm_config = get_vllm_config(config)
    with set_current_vllm_config(vllm_config):
        model_runner = ModelRunner(vllm_config)
    
    test_scenarios = []

    # --- 场景 A: Prefill 阶段 (影响 Latency ~ O(L^2)) ---
    # 固定 Batch Size，变 Sequence Length
    for L in [128, 512, 1024, 2048, 4096]:
        test_scenarios.append(TestConfig(
            name=f"Prefill_L{L}",
            batch_size=8,       # 假设并发 8 个请求
            context_len=0,
            query_len=L
        ))

    # --- 场景 B: Decode 阶段 (影响 Latency ~ O(L)) ---
    # 固定 Query=1，变 Context Length (模拟生成长文本后的延迟)
    for L in [128, 512, 1024, 2048, 4096, 8192, 16384]:
        test_scenarios.append(TestConfig(
            name=f"Decode_Ctx{L}",
            batch_size=32,      # Decode 阶段通常 Batch Size 较大
            context_len=L,
            query_len=1
        ))

    # --- 场景 C: Batch Size 扩展性 ---
    # 固定长度，变 Batch Size (用于寻找吞吐量拐点)
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        test_scenarios.append(TestConfig(
            name=f"Prefill_BS{bs}",
            batch_size=bs,
            context_len=0,
            query_len=512
        ))
        
    # --- 场景 D: 混合 / Chunked Prefill (更接近 vLLM 实际行为) ---
    # 比如 Prefill 分块执行，或者长 Context 下的短 Query
    test_scenarios.append(TestConfig(
        name="Chunked_Prefill",
        batch_size=16,
        context_len=1000,  # 之前已经有 1000 的 cache
        query_len=500      # 这一次只处理 500 个 token
    ))

    # 运行测试
    perf_data = benchmark_model_runner(
        model_runner=model_runner,
        configs=test_scenarios
    )

    # 4. (可选) 基于 perf_data 拟合简单的线性模型示例
    # 这里可以导入 sklearn 或使用 numpy 做简单的回归
    # 目标: Latency = w1 * BS * QueryLen^2 + w2 * BS * ContextLen + w3 * BS
    print("\nPerformance data collected for model fitting:")
    for row in perf_data:
        print(row)