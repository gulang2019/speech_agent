from transformers import AutoTokenizer

from vllm.config import ModelConfig, SchedulerConfig, CacheConfig, ParallelConfig, VllmConfig

from .struct import Config 
from .scheduler import Scheduler
from .engine import Engine 
from .model_runner import ModelRunner
from .kv_cache import PagedKVCacheManager

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
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
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
