import os
import torch
import math 
import numpy as np
from typing import cast, Any
import logging

logger = logging.getLogger(__name__)


from itertools import accumulate
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import init_distributed_environment, ensure_model_parallel_initialized
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


from .struct import Batch, BatchOutput

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
        return generated_tokens
