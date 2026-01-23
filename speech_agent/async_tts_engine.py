from asyncio.queues import Queue, QueueEmpty
from dataclasses import dataclass
import torch
import asyncio
import logging 
import numpy as np
import time

logger = logging.getLogger('async_stt_engine')

from moshi.models import loaders, LMGen
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel, ConditionAttributes, State
import typing as tp
from moshi.conditioners import dropout_all_conditions


@dataclass 
class RequestInstance:
    req_id: str
    input_queue: Queue
    output_queue: Queue 
    state: State 
    offset: int = -1
    input_ends: bool = False
    additional_steps: int = 0
    

def _make_null(
    all_attributes: tp.Sequence[ConditionAttributes],
) -> list[ConditionAttributes]:
    # When using CFG, returns the null conditions.
    return dropout_all_conditions(all_attributes)

class TTSEngine:
    def __init__(self,
                 model_name: str = DEFAULT_DSM_TTS_REPO,
                 voice: str = "expresso/ex03-ex01_happy_001_channel1_334s.wav",
                 device: str = 'cuda',
                 batch_size: int = 32):
        
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(model_name)
        self.tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info=checkpoint_info, 
            n_q = 32, 
            temp = 0.6, 
            device = torch.device(device)
        )
        voice_path = self.tts_model.get_voice_path(voice)
        attributes = self.tts_model.make_condition_attributes(
            [voice_path], cfg_coef=2.0
        )
        
        self.device = device
        self.mimi = checkpoint_info.get_mimi(device=device)
        self._exec_mask = [False for _ in range(batch_size)]
        self._waiting: list[RequestInstance] = []
        self._running: list[RequestInstance] = [None for _ in range(batch_size)]
        self._free_slots: list[int] = list(range(batch_size))
        self.batch_size = batch_size
        self.mimi.streaming_forever(batch_size)
        
        def _on_text_logits_hook(text_logits):
            if self.tts_model.padding_bonus:
                text_logits[..., self.tts_model.machine.token_ids.pad] += (
                    self.tts_model.padding_bonus
                )
            return text_logits
        
        def _on_audio_hook(audio_tokens):
            audio_offset = self.tts_model.lm.audio_offset
            delays = self.tts_model.lm.delays
            for bid in range(self.batch_size):
                if not self._exec_mask[bid]: continue
                req_instance = self._running[bid]
                assert req_instance is not None
                for q in range(audio_tokens.shape[1]):
                    delay = delays[q + audio_offset]
                    if req_instance.offset < delay + self.tts_model.delay_steps:
                        audio_tokens[bid, q] = self.tts_model.machine.token_ids.zero
        
        def _on_text_hook(text_tokens):
            # text_tokens has shape [B, 1, 1]
            for bid in range(self.batch_size):
                if not self._exec_mask[bid]: continue
                req_instance = self._running[bid]
                assert req_instance is not None
                token = text_tokens[bid].item()
                out_token, _ = self.tts_model.machine.process(req_instance.offset, req_instance.state, token)
                
                text_tokens[bid] = out_token
                
        assert self.tts_model.cfg_coef == 1.0, self.tts_model.cfg_coef
        assert self.tts_model.lm.condition_provider is not None
        prepared = self.tts_model.lm.condition_provider.prepare([attributes for _ in range(batch_size)])
        condition_tensors = self.tts_model.lm.condition_provider(prepared)
        
        self.tts_model.lm.dep_q = self.tts_model.n_q
        self.lm_gen = LMGen(
            self.tts_model.lm,
            temp=self.tts_model.temp,
            temp_text=self.tts_model.temp,
            cfg_coef=self.tts_model.cfg_coef,
            condition_tensors=condition_tensors,
            on_text_logits_hook=_on_text_logits_hook,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            cfg_is_masked_until=None,
            cfg_is_no_text=True,
        )
        self.lm_gen.streaming_forever(batch_size)
        
        n_delay_step = self.tts_model.delay_steps + 8 + max(self.tts_model.lm.delays)
        logger.info(f'TTS initialized w/ {model_name}')
        logger.info(f'TTS delay {n_delay_step / self.mimi.frame_rate}s, {n_delay_step} steps')
        

    @property
    def frame_size(self):
        return int(self.mimi.sample_rate / self.mimi.frame_rate)
    
    @property
    def sample_rate(self):
        return self.mimi.sample_rate
    
    def add_request(
        self,
        req_id: str,
    ) -> tuple[Queue, Queue]:
        instance = RequestInstance(
            req_id= req_id, 
            input_queue = Queue(),
            output_queue=Queue(),
            state = self.tts_model.machine.new_state([]),
            additional_steps= self.tts_model.delay_steps + 8 + max(self.tts_model.lm.delays)
        )
        
        self._waiting.append(instance)
        
        return instance.input_queue, instance.output_queue
    
    def _prepare_input(self):
        # 1. If not enough requests are in the running queue, move request from waiting to running
        while len(self._free_slots) and len(self._waiting):
            slot_id = self._free_slots.pop()
            req = self._waiting.pop()
            assert self._running[slot_id] is None and self._exec_mask[slot_id] is False 
            self._running[slot_id] = req
            self._exec_mask[slot_id] = True 

        # 2. collect the input texts
        finished = []
        for i, r in enumerate(self._running):
            if r is not None:
                if r.input_ends and not len(r.state.entries):
                    if r.additional_steps > 0:
                        r.additional_steps -= 1
                    else:
                        # the request finished
                        r.output_queue.put_nowait((time.perf_counter(), None))
                        self._running[i] = None 
                        self._exec_mask[i] = False
                        self._free_slots.append(i)
                        finished.append(i)
                        continue
                toks = []
                while not r.input_queue.empty():
                    toks.append(r.input_queue.get_nowait())
                if len(toks) and toks[-1] == '<eos>':
                    toks = toks[:-1]
                    r.input_ends = True    
                if len(toks):
                    entries = self.tts_model.prepare_script(
                        toks
                    )
                    r.state.entries.extend(entries)
                r.offset += 1
        if len(finished):
            reset_mask = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
            reset_mask[finished] = True
            self.lm_gen.reset_streaming(reset_mask)
            self.mimi.reset_streaming(reset_mask)
                
                

    def _process_response(self, frame: torch.Tensor):
        if frame is not None:
            logger.debug(f"process response of size {frame.shape}")
            assert frame.shape[2] == 1
            assert frame.shape[0] == self.batch_size
            m = (frame < 0).any(dim=(1,2))
            m = ~torch.tensor(self._exec_mask, dtype = m.dtype, device = m.device) | m
            frame[m] = 0
            m = m.cpu().tolist()
            pcm = self.mimi.decode(frame[:, 1:, :])
            if not all(m):
                for i, (m_, r) in enumerate(zip(m, self._running)):
                    if r is not None and not m_:
                        r.output_queue.put_nowait((time.perf_counter(), np.clip(pcm[i, 0].cpu(), -1, 1)))

    @torch.inference_mode()
    def _step(self):
        self._prepare_input()
        mask_t = torch.tensor(self._exec_mask, dtype = torch.bool, device = self.device)
        self.lm_gen.set_exec_mask(mask_t)
        self.mimi.set_exec_mask(mask_t)
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = torch.full(
            (self.batch_size, missing, 1),
            self.tts_model.machine.token_ids.zero,
            dtype=torch.long,
            device=self.tts_model.lm.device,
        )
        frame = self.lm_gen.step(input_tokens)
        self._process_response(frame)
    
    @property
    def on(self):
        return not (len(self._free_slots) == self.batch_size and len(self._waiting) == 0)
    
    async def run(
        self,
    ):
        logger.info('Start Engine...')
        while True: 
            if not self.on:
                await asyncio.sleep(0.05)
                continue
            self._step()
            
            await asyncio.sleep(0)

    def get_mem_usage(self):
        return self.lm_gen.lm_model.get_mem_usage()