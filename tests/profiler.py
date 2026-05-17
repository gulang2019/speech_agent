import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

import torch
import math 
import numpy as np
from typing import cast, Any
import logging
import time
import csv
import random
import matplotlib as plt
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

        self.next_block_id = 0 
        self.allocated_blocks: dict[str, list[int]] = {}
    
    def _get_or_allocate_blocks(self, req_id: str, current_total_seq_len: int) -> list[int]:
        """
        模拟VLLM的块管理：
        - 如果请求已存在 (Decode阶段)，返回已分配的块，并可能为新token分配新块
        - 如果请求不存在 (Prefill阶段)，分配新块
        """
        num_blocks_needed = (current_total_seq_len + self.block_size - 1) // self.block_size
        
        if req_id not in self.allocated_blocks:
            # Prefill阶段 或 新请求
            if self.next_block_id + num_blocks_needed > self.num_blocks:
                # 简单处理：如果块不够，就从头开始分配，但这是不真实的，
                # 真实的VLLM会淘汰旧请求或抛出OOM
                logger.warning(f"KV Cache exhausted for req {req_id}. Resetting block allocator.")
                self.next_block_id = 0 
                # 这里应该清理 self.allocated_blocks，但为了简化模拟暂时不处理
                # 如果这个警告频繁出现，说明你的 num_blocks_needed 太大或者 num_blocks 太小
            
            allocated = list(range(self.next_block_id, self.next_block_id + num_blocks_needed))
            self.next_block_id += num_blocks_needed
            self.allocated_blocks[req_id] = allocated
        else:
            # Decode阶段：请求已存在，可能需要增加一个块（如果新token跨越了块边界）
            current_allocated_blocks = self.allocated_blocks[req_id]
            if len(current_allocated_blocks) < num_blocks_needed:
                # 需要分配一个新块
                if self.next_block_id + 1 > self.num_blocks:
                    logger.warning(f"KV Cache exhausted for req {req_id} (new block). Resetting block allocator.")
                    self.next_block_id = 0 # 简单重置
                
                new_block_id = self.next_block_id
                self.next_block_id += 1
                current_allocated_blocks.append(new_block_id)
                self.allocated_blocks[req_id] = current_allocated_blocks
            allocated = self.allocated_blocks[req_id]
        
        # 确保返回的块列表长度正确
        assert len(allocated) >= num_blocks_needed, \
            f"Req {req_id}: Allocated {len(allocated)} blocks, but {num_blocks_needed} needed for seq_len {current_total_seq_len}"
        
        return allocated[:num_blocks_needed] # 截断到实际需要的块数

    def release_blocks(self, req_id: str):
        """模拟释放请求的块（可选，如果需要更真实的模拟）"""
        if req_id in self.allocated_blocks:
            del self.allocated_blocks[req_id]
        
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
        max_num_batched_tokens=10**18,
        max_model_len=10**18,
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
    model_runner: ModelRunner,
    batch_size: int,
    context_len: int,
    query_len: int,
    vocab_size: int
) -> Batch:
    batch = Batch()
    batch.req_ids = [f"req_{i}_{random.randint(1000,9999)}" for i in range(batch_size)]
    batch.context_lens = [context_len] * batch_size
    batch.query_lens = [query_len] * batch_size
    
    all_tokens = []
    for _ in range(batch_size):
        tokens = [random.randint(0, vocab_size - 1) for _ in range(query_len)]
        all_tokens.extend(tokens)
    batch.input_ids = all_tokens
    batch.block_idss = []

    for i in range(batch_size):
        req_id = batch.req_ids[i]
        total_seq_len = context_len + query_len
        # 使用模拟的块管理器来获取或分配块
        seq_blocks = model_runner._get_or_allocate_blocks(req_id, total_seq_len)
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
    print(f"{'Config Name':<20} | {'BS':<4} | {'Ctx':<6} | {'Qry':<4} | {'Latency (ms)':<12} | {'Tokens/Step':<12} | {'Tokens/s':<12} | {'CUDA memory/GB':<12}")
    print("-" * 90)

    results = [] # 存储结果供后续拟合模型使用

    for cfg in configs:
        # 1. 预热
        model_runner.next_block_id = 0
        model_runner.allocated_blocks = {} # 清空已分配的块
        for _ in range(num_warmup):
            warmup_batch = create_mock_batch(
                model_runner,cfg.batch_size, cfg.context_len, cfg.query_len, vocab_size
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
                model_runner,cfg.batch_size, cfg.context_len, cfg.query_len, vocab_size
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
        
        print(f"{cfg.name:<20} | {cfg.batch_size:<4} | {cfg.context_len:<6} | {cfg.query_len:<4} | {avg_time_ms:<12.2f} | {avg_tokens_per_step:<12.0f} | {throughput_tokens:<12.0f} | {(torch.cuda.memory_allocated() / 1024**3):<12.0f}")

        # 保存结果用于性能模型拟合
        results.append({
            "batch_size": cfg.batch_size,
            "context_len": cfg.context_len,
            "query_len": cfg.query_len,
            "latency_ms": avg_time_ms,
            "throughput": throughput_tokens
        })
    
    return results

def separate_prefill_decode(perf_data: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    将性能数据分为 Prefill 和 Decode 两组
    """
    prefill_data = [row for row in perf_data if row['context_len'] == 0]
    decode_data = [row for row in perf_data if row['query_len'] == 1]
    return prefill_data, decode_data

def prepare_prefill_features(prefill_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    准备 Prefill 的特征矩阵 X 和目标值 y
    理论模型：Latency ≈ α * (B * Q²) + β * (B * Q) + γ
    """
    X = []
    y = []
    
    for row in prefill_data:
        B = row['batch_size']
        Q = row['query_len']
        latency = row['latency_ms']
        
        # 特征1：B * Q²  (主导项，来自 Attention 的二次复杂度)
        # 特征2：B * Q    (线性项，来自其他操作)
        feature1 = B * (Q ** 2)
        feature2 = B * Q
        
        X.append([feature1, feature2])
        y.append(latency)
    
    return np.array(X), np.array(y)

def prepare_decode_features(decode_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    准备 Decode 的特征矩阵 X 和目标值 y
    理论模型：Latency ≈ α * (B * C) + β * B + γ
    """
    X = []
    y = []
    
    for row in decode_data:
        B = row['batch_size']
        C = row['context_len']
        latency = row['latency_ms']
        
        # 特征1：B * C  (主导项，来自 KV Cache 读取)
        # 特征2：B     (固定开销)
        feature1 = B * C
        feature2 = B
        
        X.append([feature1, feature2])
        y.append(latency)
    
    return np.array(X), np.array(y)

# ==================== 回归模型 ====================

def linear_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    使用 NumPy 进行线性回归 (最小二乘法)
    返回：系数, 截距, R²
    """
    # 添加偏置项 (列全为1)
    X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
    
    # 最小二乘法求解: (X^T X)^(-1) X^T y
    coeffs, residuals, rank, s = np.linalg.lstsq(X_with_bias, y, rcond=None)
    
    # 计算预测值
    y_pred = X_with_bias @ coeffs
    
    # 计算 R²
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # 提取系数和截距
    w = coeffs[:-1]
    bias = coeffs[-1]
    
    return w, bias, r_squared

# ==================== 可视化 ====================

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    """
    绘制预测值 vs 真实值的散点图
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
    
    # 绘制完美预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Latency (ms)', fontsize=12)
    plt.ylabel('Predicted Latency (ms)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_names: list[str], coeffs: np.ndarray, title: str):
    """
    绘制特征重要性 (系数绝对值)
    """
    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, np.abs(coeffs), color='steelblue', edgecolor='k')
    plt.ylabel('Coefficient Magnitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# ==================== 主程序 ====================

def build_performance_model(perf_data: list[dict]):
    """
    构建性能模型的主函数
    """
    print("=" * 80)
    print("Performance Model Building")
    print("=" * 80)
    
    # 1. 分离数据
    prefill_data, decode_data = separate_prefill_decode(perf_data)
    print(f"\n📊 数据统计:")
    print(f"  - Prefill 样本数: {len(prefill_data)}")
    print(f"  - Decode 样本数: {len(decode_data)}")
    
    # ==================== Prefill 模型 ====================
    if len(prefill_data) > 0:
        print("\n" + "=" * 80)
        print("🔥 Prefill 模型 (Context = 0)")
        print("=" * 80)
        
        X_prefill, y_prefill = prepare_prefill_features(prefill_data)
        w_prefill, bias_prefill, r2_prefill = linear_regression(X_prefill, y_prefill)
        
        # 打印结果
        print(f"\n✅ 回归结果 (R² = {r2_prefill:.4f}):")
        print(f"   Latency (ms) = {w_prefill[0]:.6e} * (B * Q²)")
        print(f"                 + {w_prefill[1]:.6e} * (B * Q)")
        print(f"                 + {bias_prefill:.4f}")
        
        # 预测
        y_pred_prefill = np.column_stack([X_prefill, np.ones(X_prefill.shape[0])]) @ np.append(w_prefill, bias_prefill)
        
    #     # 可视化
    #     plot_predictions(
    #         y_prefill, 
    #         y_pred_prefill, 
    #         f'Prefill Model: Predicted vs Actual (R² = {r2_prefill:.4f})'
    #     )
    #     plot_feature_importance(
    #         ['B * Q²', 'B * Q'], 
    #         w_prefill, 
    #         'Prefill Model: Feature Importance'
    #     )
    # else:
        print("\n⚠️  没有 Prefill 数据，跳过 Prefill 模型")
    
    # ==================== Decode 模型 ====================
    if len(decode_data) > 0:
        print("\n" + "=" * 80)
        print("⚡ Decode 模型 (Query = 1)")
        print("=" * 80)
        
        X_decode, y_decode = prepare_decode_features(decode_data)
        w_decode, bias_decode, r2_decode = linear_regression(X_decode, y_decode)
        
        # 打印结果
        print(f"\n✅ 回归结果 (R² = {r2_decode:.4f}):")
        print(f"   Latency (ms) = {w_decode[0]:.6e} * (B * C)")
        print(f"                 + {w_decode[1]:.6e} * B")
        print(f"                 + {bias_decode:.4f}")
        
        # 预测
        y_pred_decode = np.column_stack([X_decode, np.ones(X_decode.shape[0])]) @ np.append(w_decode, bias_decode)
        
        # # 可视化
        # plot_predictions(
        #     y_decode, 
        #     y_pred_decode, 
        #     f'Decode Model: Predicted vs Actual (R² = {r2_decode:.4f})'
        # )
        # plot_feature_importance(
        #     ['B * C', 'B'], 
        #     w_decode, 
        #     'Decode Model: Feature Importance'
        # )
    else:
        print("\n⚠️  没有 Decode 数据，跳过 Decode 模型")
    
    # ==================== 预测函数 ====================
    def predict_latency(batch_size: int, context_len: int, query_len: int) -> float:
        """
        使用拟合的模型预测延迟
        """
        if context_len == 0 and len(prefill_data) > 0:
            # Prefill
            x = np.array([[batch_size * query_len**2, batch_size * query_len]])
            return float((np.column_stack([x, np.ones(1)]) @ np.append(w_prefill, bias_prefill))[0])
        elif query_len == 1 and len(decode_data) > 0:
            # Decode
            x = np.array([[batch_size * context_len, batch_size]])
            return float((np.column_stack([x, np.ones(1)]) @ np.append(w_decode, bias_decode))[0])
        else:
            # 混合情况 (简单相加)
            prefill_time = 0
            decode_time = 0
            
            if len(prefill_data) > 0 and query_len > 0:
                x = np.array([[batch_size * query_len**2, batch_size * query_len]])
                prefill_time = float((np.column_stack([x, np.ones(1)]) @ np.append(w_prefill, bias_prefill))[0])
            
            if len(decode_data) > 0:
                x = np.array([[batch_size * context_len, batch_size]])
                decode_time = float((np.column_stack([x, np.ones(1)]) @ np.append(w_decode, bias_decode))[0])
            
            return prefill_time + decode_time
    
    print("\n" + "=" * 80)
    print("🎯 预测示例")
    print("=" * 80)
    
    examples = [
        (8, 0, 128),    # Prefill
        (16, 2048, 1),  # Decode
        (4, 500, 500),  # Chunked
    ]
    
    for B, C, Q in examples:
        pred = predict_latency(B, C, Q)
        print(f"  Batch={B}, Context={C}, Query={Q} → Predicted Latency: {pred:.2f} ms")
    
    return predict_latency

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
    for L in [64,128, 192,256,384,512]:
        test_scenarios.append(TestConfig(
            name=f"Prefill_L{L}",
            batch_size=8,       # 假设并发 8 个请求
            context_len=0,
            query_len=L
        ))

    # --- 场景 B: Decode 阶段 (影响 Latency ~ O(L)) ---
    # 固定 Query=1，变 Context Length (模拟生成长文本后的延迟)
    for L in [64,128,256,512,1024]:
        test_scenarios.append(TestConfig(
            name=f"Decode_Ctx{L}",
            batch_size=4,      # Decode 阶段通常 Batch Size 较大
            context_len=L,
            query_len=1
        ))

    # --- 场景 C: Batch Size 扩展性 ---
    # 固定长度，变 Batch Size (用于寻找吞吐量拐点)
    for bs in [1, 2, 4, 8, 16]:
        test_scenarios.append(TestConfig(
            name=f"Prefill_BS{bs}",
            batch_size=bs,
            context_len=128,
            query_len=1
        ))


    # 运行测试
    perf_data = benchmark_model_runner(
        model_runner=model_runner,
        configs=test_scenarios
    )

    predictor = build_performance_model(perf_data)

    val_times = 5

    for i in range(val_times):
        val_scenarios = []
        bs = random.randint(1,16)
        cl = random.randint(16,512)
        ql = random.randint(1,32)

        val_scenarios.append(TestConfig(
            name=f"Chunked_Prefill_{i}",
            batch_size=bs,
            context_len=cl,  # 之前已经有 1000 的 cache
            query_len=ql      # 这一次只处理 500 个 token
        ))
        val_data = benchmark_model_runner(
            model_runner=model_runner,
            configs=val_scenarios
        )
        al = val_data[0]["latency_ms"]
        print(f"Actual Latency:{al}")
        print(f"Predicted Latency:{predictor(batch_size=bs, context_len=cl, query_len=ql)}")

    
    # 4. (可选) 基于 perf_data 拟合简单的线性模型示例
    # 这里可以导入 sklearn 或使用 numpy 做简单的回归
    # 目标: Latency = w1 * BS * QueryLen^2 + w2 * BS * ContextLen + w3 * BS
    # print("\nPerformance data collected for model fitting:")
    # for row in perf_data:
    #     print(row)