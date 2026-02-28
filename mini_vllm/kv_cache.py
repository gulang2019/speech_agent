import math
from dataclasses import dataclass, field
import heapq
import itertools
from typing import Any, Callable, Generic, Optional, TypeVar, Iterator

from .struct import KVCacheBlock, Request

_HASH_TYPE = str
def DEFAULT__hashER(block_ids: list[int]) -> _HASH_TYPE:
    # Stable, deterministic hash for a block.
    return str(tuple(block_ids))

@dataclass 
class LRUPrefixCache:
    _n_freed: int = 0
    _root_block: KVCacheBlock = field(init = False)
    _head: KVCacheBlock = field(init = False)
    _tail: KVCacheBlock = field(init = False)
        
    def __post_init__(self):
        self._root_block = KVCacheBlock(idx = -1, reference_cnt = 0)
        # we use a baseline to always evict the deepest leaves. 
        self._head = KVCacheBlock(idx = -1)
        self._tail = KVCacheBlock(idx = -1)
        self._head.next = self._tail 
        self._tail.prev = self._head
        
    def get_max_allocable(self, hashes: list[_HASH_TYPE]):
        _block = self._root_block 
        n_allocable = self._n_freed 
        for hash in hashes:
            if hash in _block.next_blocks:
                _block = _block.next_blocks[hash]
                if _block.reference_cnt > 0: 
                    n_allocable += 1
            else: break 
        return n_allocable
                    
        
    def allocate(self, hashes: list[_HASH_TYPE]):
        _block = self._root_block 
        allocated = []
        for hash in hashes:
            if hash in _block.next_blocks:
                _block = _block.next_blocks[hash]
                assert _block._hash == hash
                allocated.append(_block)
                if _block.reference_cnt == 0:  # remove the block from the linked list of free requests 
                    assert _block.prev is not None 
                    assert _block.next is not None
                    _block.prev.next = _block.next
                    _block.next.prev = _block.prev 
                    _block.prev = _block.next = None
                    self._n_freed -= 1
                _block.reference_cnt += 1
            else: break
        return allocated
    
    def free(self, blocks: list[KVCacheBlock]):
        # free computed blocks
        prev_block = self._root_block
        push_at_block = self._head
        # we assume that the blocks are from root to leaves
        for block in blocks: 
            prev_block.next_blocks[block._hash] = block
            block.parent = prev_block
            block.reference_cnt -= 1 
            if block.reference_cnt == 0:
                # we push to the front
                push_at_block.next.prev = block 
                block.next = push_at_block.next 
                block.prev = push_at_block 
                push_at_block.next = block 
                push_at_block = block
                self._n_freed += 1
            prev_block = block
            
    def evict(self, n: int) -> Optional[list[KVCacheBlock]]:
        # evict from the end
        if not self._n_freed >= n: return None 
        block = self._tail.prev 
        blocks = []
        for _ in range(n):
            assert block is not self._head
            assert len(block.next_blocks) == 0
            self._n_freed -= 1 
            blocks.append(block)
            block.next.prev = block.prev 
            block.prev.next = block.next 
            next_block = block.prev
            block.next = block.prev = None 
            block.parent.next_blocks.pop(block._hash)
            block.parent = None 
            block._hash = None
            block = next_block
        return blocks
        
@dataclass
class PagedKVCacheManager:
    block_size: int
    num_blocks: int
    _free_blocks: list[KVCacheBlock] = field(default_factory=list)
    _prefix_cache: LRUPrefixCache = field(default_factory=LRUPrefixCache)
    __hasher: Callable[[list[int]], _HASH_TYPE] = DEFAULT__hashER
    
    @staticmethod 
    def from_vllm_config(vllm_config):
        from vllm.config import VllmConfig 
        assert isinstance(vllm_config, VllmConfig)
        return PagedKVCacheManager(
            
        )
    
    def __post_init__(self):
        # kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
        # we prioritize request not on tree; for request on tree, we prioritize leaves 
        for idx in range(self.num_blocks):
            self._free_blocks.append(KVCacheBlock(idx = idx))
    
    def _get_num_needed_blocks(self, n_tokens: int):
        return math.ceil(n_tokens / self.block_size)
    
    def _hash(self, tokens: list[int]):
        hashes = []    
        for i in range(self.block_size, len(tokens) + 1, self.block_size):
            hash = self.__hasher(tokens[i-self.block_size:i])
            hashes.append(hash)
        return hashes 
    
    def _alloc_from_free_and_evict(self, n):
        assert n >= 0
        assert n <= len(self._free_blocks) + self._prefix_cache._n_freed
        n_from_freed = min(len(self._free_blocks), n)
        n_from_evict = n - n_from_freed
        free_blocks = self._free_blocks[:n_from_freed]
        self._free_blocks = self._free_blocks[n_from_freed:]
        evict_blocks = self._prefix_cache.evict(n_from_evict)
        assert evict_blocks is not None 
        new_blocks = free_blocks + evict_blocks
        for b in new_blocks:
            assert b.reference_cnt == 0 
            assert b._hash is None 
            assert b.parent is None 
            assert len(b.next_blocks) == 0
            assert b.prev is None and b.next is None 
            b.reference_cnt += 1 
        return new_blocks
    
    def allocate_prefill(self, request: Request) -> bool:
        assert request.n_computed_tokens == 0
        assert request.blocks is None
        prompt_hashes = self._hash(request.prompt_ids)
        
        n = self._get_num_needed_blocks(request.n_prompt_tokens)
        
        if self._prefix_cache.get_max_allocable(prompt_hashes) + len(self._free_blocks) < n:
            return False 
        
        cached_blocks = self._prefix_cache.allocate(prompt_hashes)
        new_blocks = self._alloc_from_free_and_evict(n - len(cached_blocks))
        
        request.blocks = cached_blocks + new_blocks 
        request.n_computed_tokens = len(cached_blocks) * self.block_size
        return True
    
    def allocate(self, request: Request, n_new_tokens: int) -> bool:
        
        n_blocks_needed = self._get_num_needed_blocks(n_new_tokens + request.n_computed_tokens) - request.n_blocks
        
        if n_blocks_needed <= 0:
            return True
        if len(self._free_blocks) + self._prefix_cache._n_freed < n_blocks_needed:
            return False 
        new_blocks = self._alloc_from_free_and_evict(n_blocks_needed)
        request.blocks.extend(new_blocks)
        return True
        
    def free(self, request: Request):
        computed_hashes = self._hash(request.computed_tokens)
        # put every full block into the prefix cache; 
        for block, hash in zip(request.blocks, computed_hashes):
            assert block._hash is None or block._hash == hash 
            block._hash = hash 
        self._prefix_cache.free(request.blocks[:len(computed_hashes)])
        for b in request.blocks[len(computed_hashes):]:
            assert b._hash is None
            self._free_blocks.append(b)
            b.reference_cnt -= 1 
            assert b.reference_cnt == 0
        # request.blocks.clear()
        # put incomplete block to the free list 