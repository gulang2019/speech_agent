from asyncio.queues import Queue, QueueEmpty
from dataclasses import dataclass
import torch
import asyncio
import logging 
import time

logger = logging.getLogger('async_stt_engine')

from moshi.models import loaders, MimiModel, LMModel, LMGen

@dataclass 
class RequestInstance:
    req_id: str 
    input_queue: Queue
    output_queue: Queue 

class STTEngine:
    def __init__(self,
                 model_name: str = "kyutai/stt-2.6b-en",
                 device: str = 'cuda',
                 batch_size: int = 32):
        
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(model_name)
        self.stt_config = checkpoint_info.stt_config
        
        self.device = device
        self.mimi = checkpoint_info.get_mimi(device=device)
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()
        self.lm = checkpoint_info.get_moshi(device = device)
        self._waiting: list[RequestInstance] = []
        self._running: list[RequestInstance] = [None for _ in range(batch_size)]
        self._free_slots: list[int] = list(range(batch_size))
        self._running_slots: list[int] = []
        self.batch_size = batch_size
        self._zero = torch.zeros(size = (1,1,self.frame_size), dtype = torch.float32, device = device)
        self.lm_gen = LMGen(self.lm, temp = 0, temp_text = 0, use_sampling = False)
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)
        
        stt_delay_padding = (self.stt_config.get("audio_delay_seconds", 0.0) + 1.0)
        stt_silence_prefix_padding = self.stt_config.get("audio_silence_prefix_seconds", 0.0)
        self.n_left_pad_frame = int(stt_silence_prefix_padding * self.mimi.frame_rate)
        self.n_right_pad_frame = int(stt_delay_padding * self.mimi.frame_rate)
        logger.info(f"STT initialized w/ {model_name}")
        logger.info(f"STT silence {stt_silence_prefix_padding}s, {self.n_left_pad_frame} FWD steps")
        logger.info(f"STT delayed {stt_delay_padding}s, {self.n_right_pad_frame} FWD steps")

    @property
    def frame_size(self):
        return int(self.mimi.sample_rate / self.mimi.frame_rate)
    
    @property 
    def delay(self):
        return 1 / self.mimi.frame_rate
    
    def add_request(
        self,
        req_id: str,
    ):
        instance = RequestInstance(
            req_id= req_id, 
            input_queue = Queue(),
            output_queue=Queue()
        )
        
        for _ in range(self.n_left_pad_frame):
            instance.input_queue.put_nowait(self._zero)
        
        self._waiting.append(instance)
        return instance.input_queue, instance.output_queue
    
    def _prepare_input(self):
        # 1. If not enough requests are in the running queue, move request from waiting to running
        while len(self._free_slots) and len(self._waiting):
            slot_id = self._free_slots.pop()
            req = self._waiting.pop()
            self._running[slot_id] = req

        # 2. collect the input pcms 
        pcms = []
        mask = []
        finished = []
        for i, r in enumerate(self._running):
            pcm = None
            if r is not None:
                try:
                    msg = r.input_queue.get_nowait()
                    
                    if isinstance(msg, str):
                        if msg == '<eos>':
                            for _ in range(self.n_right_pad_frame):
                                self._running[i].input_queue.put_nowait(self._zero)
                            self._running[i].input_queue.put_nowait('<eoa>')
                        elif msg == '<eoa>': # end of asr
                            self._running[i].output_queue.put_nowait((time.perf_counter(), '<eoa>'))
                            self._running[i] = None 
                            self._free_slots.append(i)
                            finished.append(i)
                    else:
                        # [1, 1, frame_size]
                        assert isinstance(msg, torch.Tensor) and msg.shape[-1] == self.frame_size
                        pcm = msg.to(self.device)
                except QueueEmpty as e:
                    pass
            mask.append(pcm is not None)
            pcms.append(pcm if pcm is not None else self._zero)
        
        if len(finished):
            reset_mask = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
            reset_mask[finished] = True
            self.mimi.reset_streaming(reset_mask)
            self.lm_gen.reset_streaming(reset_mask)
        
        if not any(mask):
            return None, mask
        in_pcm = torch.cat(pcms, dim = 0)
        # shape B, S, 1
        return in_pcm, mask
    
    def _process_response(self, tokens: torch.Tensor, mask: list[bool]):
        assert tokens.shape[1] == 1

        tensor = tokens.cpu()
        for i, m in enumerate(mask):
            if not m: continue
            assert self._running[i] is not None 
            token = tensor[i, 0].item()
            if token not in [0, 3]:
                text = self.text_tokenizer.id_to_piece(token)
                text = text.replace("▁", " ")
            else: 
                text = '<PAD>'
            self._running[i].output_queue.put_nowait((time.perf_counter(), text))
    
    @torch.inference_mode()
    async def run(
        self,
    ):
        logger.info('Start Engine...')
        while True: 
            if len(self._free_slots) == self.batch_size and len(self._waiting) == 0:
                await asyncio.sleep(0.05)
                continue

            in_pcm, mask = self._prepare_input()
            
            if in_pcm is None:
                await asyncio.sleep(0)
                continue
            
            mask_t = torch.tensor(mask, dtype = torch.bool, device = self.device)
            self.lm_gen.set_exec_mask(mask_t)
            self.mimi.set_exec_mask(mask_t)
            encoded = self.mimi.encode(in_pcm)
            tokens = self.lm_gen.step(encoded)
            if tokens is None:
                await asyncio.sleep(0)
                continue
            self._process_response(tokens, mask)
            await asyncio.sleep(0)            

    def get_mem_usage(
      self  
    ):
        return self.lm.get_mem_usage()