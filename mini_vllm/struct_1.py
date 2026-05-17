from dataclasses import field, dataclass 
from typing import Optional
from asyncio.queues import Queue
@dataclass 
class Request:
    req_id: str
    prompt_text: str
    prompt_ids: Optional[list[int]] = None
    max_tokens: int | None = None
    ignore_eos: bool = True

    output_queue: Queue = field(default_factory=Queue)
    
    # the number of tokens computed
    n_computed_tokens: int = 0
    blocks: Optional[list['KVCacheBlock']] = None 
    generated_ids: list[int] = field(default_factory=list, init=False)

    @property
    def finished_prefill(self):
        return self.n_computed_tokens >= len(self.prompt_ids)

    @property
    def n_prompt_tokens(self):
        return len(self.prompt_ids)
    
    @property
    def n_generated_tokens(self):
        return len(self.generated_ids)

    @property 
    def n_blocks(self):
        return len(self.blocks)
    
    @property 
    def block_ids(self) -> list[int]:
        return [block.idx for block in self.blocks]

    @property 
    def computed_tokens(self) -> list[int]:
        return (self.prompt_ids + self.generated_ids)[:self.n_computed_tokens]
    
    @property 
    def check_invariants(self):
        if self.n_computed_tokens < len(self.prompt_ids):
            assert len(self.generated_ids) == 0
        else:
            assert self.n_computed_tokens == (len(self.prompt_ids) + len(self.generated_ids) - 1)

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
        return sum([list(range(cl, cl + ql)) for cl, ql in zip(self.context_lens, self.query_lens)], start = [])
    
    @property
    def seq_lens(self) -> list[int]:
        return [cl + ql for cl, ql in zip(self.context_lens, self.query_lens)]
    
BatchOutput = dict[str, list[int]]


@dataclass
class KVCacheBlock:
    idx: int
    reference_cnt: int = 0
    _hash: Optional[str] = None
    parent: Optional["KVCacheBlock"] = None
    next_blocks: dict[str, "KVCacheBlock"] = field(default_factory=dict)
    prev: Optional['KVCacheBlock'] = None 
    next: Optional['KVCacheBlock'] = None 


@dataclass 
class Config:
    model_name: str
    max_memory_utilization: float 
    block_size: int = 16
