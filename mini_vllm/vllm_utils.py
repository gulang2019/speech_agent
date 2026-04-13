from ._bootstrap import bootstrap_vllm_import_env

bootstrap_vllm_import_env()

from transformers import AutoTokenizer

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.compilation import CUDAGraphMode, CompilationConfig, CompilationMode

from .struct import Config
from .scheduler import Scheduler
from .engine import Engine
from .model_runner import ModelRunner
from .kv_cache import PagedKVCacheManager


_COMPILATION_MODE_MAP = {
    "none": CompilationMode.NONE,
    "stock_torch_compile": CompilationMode.STOCK_TORCH_COMPILE,
    "dynamo_trace_once": CompilationMode.DYNAMO_TRACE_ONCE,
    "vllm_compile": CompilationMode.VLLM_COMPILE,
}

_CUDAGRAPH_MODE_MAP = {
    "none": CUDAGraphMode.NONE,
    "piecewise": CUDAGraphMode.PIECEWISE,
    "full": CUDAGraphMode.FULL,
    "full_decode_only": CUDAGraphMode.FULL_DECODE_ONLY,
    "full_and_piecewise": CUDAGraphMode.FULL_AND_PIECEWISE,
}

def get_vllm_config(
    config: Config
):
    max_num_batched_tokens = (
        config.max_num_batched_tokens
        if config.max_num_batched_tokens is not None
        else 512
    )
    model_config = ModelConfig(
        model=config.model_name,
        dtype="float16",
        seed=42,
        enforce_eager=config.enforce_eager,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=max_num_batched_tokens,
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
    compilation_config = None
    if (
        config.compilation_mode is not None
        or config.compilation_backend
        or config.cudagraph_mode is not None
    ):
        compilation_kwargs = {}
        if config.compilation_mode is not None:
            compilation_kwargs["mode"] = _COMPILATION_MODE_MAP[config.compilation_mode]
        if config.compilation_backend:
            compilation_kwargs["backend"] = config.compilation_backend
        if config.cudagraph_mode is not None:
            compilation_kwargs["cudagraph_mode"] = _CUDAGRAPH_MODE_MAP[
                config.cudagraph_mode
            ]
        compilation_config = CompilationConfig(**compilation_kwargs)
    vllm_kwargs = {}
    if compilation_config is not None:
        vllm_kwargs["compilation_config"] = compilation_config
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        **vllm_kwargs,
    )
    return vllm_config


def get_engine_from_vllm(
    config: Config 
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    vllm_config = get_vllm_config(config)
    model_runner = ModelRunner(vllm_config)
    
    memory_manager = PagedKVCacheManager(
        config.block_size,
        num_blocks=model_runner.num_blocks
    )
    
    scheduler = Scheduler(memory_manager = memory_manager, 
                          eos_token_id = tokenizer.eos_token_id)

    engine = Engine(tokenizer, scheduler, model_runner)
    
    return engine 
    
# vllm_config = get_vllm_config()
