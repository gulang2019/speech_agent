from typing import Optional
import asyncio
import torch 
import numpy as np
from dataclasses import dataclass, field, asdict
import time 
from datasets import load_dataset
import tiktoken
from scipy.signal import resample_poly
from pathlib import Path
import json
import logging
import pandas as pd
import tqdm
import math
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, List, Any, Callable

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from speech_agent.async_stt_engine import STTEngine
from speech_agent.async_tts_engine import TTSEngine
from vllm_utils import VLLMConfig, vllm_chat_stream

now = time.perf_counter

stt_engine: Optional[STTEngine] = None 
tts_engine: Optional[TTSEngine] = None 
batch_size: int = 16
stt_engine_task: Optional[asyncio.Task] = None
tts_engine_task: Optional[asyncio.Task] = None
vllm_config: Optional[VLLMConfig] = None 

def on_done(t: asyncio.Task):
    try:
        t.result()   # re-raises exception if task failed
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print("Background task failed", e)

async def _init():
    global stt_engine, tts_engine, stt_engine_task, tts_engine_task, vllm_config
    if stt_engine is not None: return 
    stt_engine = STTEngine(batch_size=batch_size)
    tts_engine = TTSEngine(batch_size = batch_size)
    stt_engine_task = asyncio.create_task(stt_engine.run())
    stt_engine_task.add_done_callback(on_done)
    tts_engine_task = asyncio.create_task(tts_engine.run())
    tts_engine_task.add_done_callback(on_done)
    vllm_config = VLLMConfig()

Sampler = Callable[[], Any]  # returns value or (used, total)

@dataclass
class UsageMonitor:
    interval: float = 1e-2
    samplers: Dict[str, Sampler] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    _stop: Optional[asyncio.Event] = None
    _task: Optional[asyncio.Task] = None
    _t0: float = 0.0

    async def __aenter__(self):
        self._stop = asyncio.Event()
        self._t0 = time.time()
        self._task = asyncio.create_task(self._run(), name="usage-monitor")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    async def stop(self):
        if not self._task:
            return
        self._stop.set()
        try:
            await self._task
        finally:
            self._task = None
            self._stop = None

    async def _run(self):
        while not self._stop.is_set():
            t = time.time()
            rec = {"timestamp": t, "t_rel": t - self._t0}
            for name, fn in self.samplers.items():
                v = fn()
                if isinstance(v, tuple) and len(v) == 2:
                    rec[name], rec[f"{name}_tot"] = v
                else:
                    rec[name] = v
            self.results.append(rec)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass


act_counts = {"stt": 0, "lm": 0, "tts": 0}
_act_lock = asyncio.Lock()

async def act_event(act: str, kind: str):
    # kind: "start" | "end"
    async with _act_lock:
        act_counts[act] += (1 if kind == "start" else -1)

# (A) NVML SM utilization (recommended if available)
def make_nvml_sm_sampler(gpu_index: int = 0) -> Sampler:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return lambda: int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)  # percent
    
def make_kv_cache_sampler(url="http://localhost:8000/v1/kv_cache_usage_history",
                          engine=0, seconds=1.0, api_key="tok-123"):
    import json, urllib.request, urllib.parse

    last_ts = 0.0

    def sample():
        nonlocal last_ts
        qs = urllib.parse.urlencode({"seconds": seconds, "engine": engine})
        req = urllib.request.Request(f"{url}?{qs}")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        data = json.loads(urllib.request.urlopen(req).read().decode("utf-8", "ignore"))
        samples = [s for s in data.get("samples", []) if s.get("ts", 0) > last_ts]
        if samples:
            last_ts = samples[-1]["ts"]
        return samples

    return sample

@dataclass 
class Script:
    audio: np.ndarray
    speaker: str 
    text: str
    n_tokens: int 

@dataclass 
class Request:    
    req_id: str
    scripts: list[Script]
    
@dataclass
class SampleSeries:
    xs: List[float] = field(default_factory=list)

    def add(self, x: Optional[float]):
        if x is None:
            return
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return
        self.xs.append(x)

    def summary(self) -> Dict[str, float]:
        keys = ["min", "p20", "p50", "p75", "p99", "max", "mean", "std", "n"]
        if not self.xs:
            out = {k: float("nan") for k in keys}
            out["n"] = 0
            return out
        a = np.asarray(self.xs, dtype=np.float64)
        return {
            "n": int(a.size),
            "min": float(a.min()),
            "p20": float(np.percentile(a, 20)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=0)),
        }

    def _ms_mean_std(self):
        if not self.xs:
            return None
        a = np.asarray(self.xs, dtype=np.float64) * 1e3
        return float(a.mean()), float(a.std(ddof=0))


def _fmt_ms(s: SampleSeries, prec: int = 1) -> str:
    ms = s._ms_mean_std()
    if ms is None:
        return "—"
    m, sd = ms
    return f"{m:.{prec}f}±{sd:.{prec}f}ms"


def _fmt_s(s: SampleSeries, prec: int = 2) -> str:
    if not s.xs:
        return "—"
    a = np.asarray(s.xs, dtype=np.float64)
    return f"{a.mean():.{prec}f}±{a.std(ddof=0):.{prec}f}s"


@dataclass
class ChatTimingAgg:
    # per-turn
    user_speak: SampleSeries = field(default_factory=SampleSeries)
    stt_delay: SampleSeries = field(default_factory=SampleSeries)
    lm_ttft: SampleSeries = field(default_factory=SampleSeries)
    lm_decode: SampleSeries = field(default_factory=SampleSeries)
    tts_delay: SampleSeries = field(default_factory=SampleSeries)
    ttff_total: SampleSeries = field(default_factory=SampleSeries)  # user_stop -> first_pcm

    # per-token / per-frame / per-chunk cadence
    stt_per_frame: SampleSeries = field(default_factory=SampleSeries)
    lm_tpot: SampleSeries = field(default_factory=SampleSeries)
    frame_dt: SampleSeries = field(default_factory=SampleSeries)

    def one_liner(self) -> str:
        return (
            f"Spk {_fmt_s(self.user_speak)} "
            f"STT {_fmt_ms(self.stt_delay)}/{_fmt_ms(self.stt_per_frame)} | "
            f"LM ttft {_fmt_ms(self.lm_ttft)} tpot {_fmt_ms(self.lm_tpot)} "
            f"dec {_fmt_s(self.lm_decode)} | "
            f"TTS {_fmt_ms(self.tts_delay)}/{_fmt_ms(self.frame_dt)} "
            f"TTFF {_fmt_ms(self.ttff_total)}"
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "user_speak": self.user_speak.summary(),
            "stt_per_frame": self.stt_per_frame.summary(),
            "stt_delay": self.stt_delay.summary(),
            "lm_ttft": self.lm_ttft.summary(),
            "lm_tpot": self.lm_tpot.summary(),
            "lm_decode": self.lm_decode.summary(),
            "tts_delay": self.tts_delay.summary(),
            "frame_dt": self.frame_dt.summary(),
            "ttff_total": self.ttff_total.summary(),
        }

async def process_request(req: Request):
    messages = [{"role": "system", "content": vllm_config.system_prompt}]
    audio_outputs = []


    agg = ChatTimingAgg()

    tqdm_construct = tqdm.tqdm(
        range(0, len(req.scripts), 2),
        desc=f"running {req.req_id[:10]}",
        dynamic_ncols=True,
    )

    for rid in tqdm_construct:
        stt_in, stt_out = stt_engine.add_request(req_id=req.req_id + f"-{rid}")
        audio = req.scripts[rid].audio
        assert req.scripts[rid].speaker == "human"
        assert rid + 1 < len(req.scripts) and req.scripts[rid + 1].speaker == "gpt"

        # -----------------------
        # Stage 1: user speaking
        # -----------------------
        await act_event(act = 'stt', kind = 'start')
        t_user_start = now()
        for i in range(0, audio.shape[0], stt_engine.frame_size):
            if i + stt_engine.frame_size >= audio.shape[0]:
                break
            await asyncio.sleep(stt_engine.delay)
            chunk = torch.Tensor(audio[i:i + stt_engine.frame_size]).reshape([1, 1, -1])
            stt_in.put_nowait(chunk)

        stt_in.put_nowait("<eos>")
        t_user_stop = now()
        agg.user_speak.add(t_user_stop - t_user_start)

        # STT delay: user stops -> STT finishes (<eoa>)
        texts = []
        last_ts = t_user_start
        while True:
            ts, txt = await stt_out.get()
            agg.stt_per_frame.add(ts - last_ts)
            last_ts = ts
            if txt == "<eoa>":
                t_stt_done = now()
                break
            if txt != '<PAD>':
                texts.append(txt)
        agg.stt_delay.add(t_stt_done - t_user_stop)
        await act_event(act = 'stt', kind = 'end')

        # keep your original arrays if you still want them

        # -----------------------
        # Stage 2: LM + TTS
        # -----------------------
        await act_event(act = 'lm', kind = 'start')
        await act_event(act = 'tts', kind = 'start')
        msg = "".join(texts)
        messages.append({"role": "user", "content": msg})

        tts_in, tts_out = tts_engine.add_request(req_id=req.req_id + f"-{rid}")
        pcms = []
        stopped_tts = False
        text_outputs = []

        t_lm_start = now()
        t_first_tok: Optional[float] = None
        t_last_tok: Optional[float] = None
        prev_tok_t: Optional[float] = None

        t_first_tts_text: Optional[float] = None
        t_first_pcm: Optional[float] = None
        prev_frame_t: Optional[float] = None

        async for text in vllm_chat_stream(
            vllm_config, messages,
            max_tokens=req.scripts[rid + 1].n_tokens,
            ignore_eos=True,
        ):
            t_tok = now()
            text_outputs.append(text)

            # LM TTFT + TPOT
            if t_first_tok is None:
                t_first_tok = t_tok
                agg.lm_ttft.add(t_first_tok - t_lm_start)
            else:
                if prev_tok_t is not None:
                    agg.lm_tpot.add(t_tok - prev_tok_t)
            prev_tok_t = t_tok
            t_last_tok = t_tok

            # feed TTS + measure first-text->first-pcm
            if t_first_tts_text is None:
                t_first_tts_text = t_tok
            tts_in.put_nowait(text)

            # drain available audio frames and record frame intervals
            while not tts_out.empty():
                t_frame, pcm = tts_out.get_nowait()
                # t_frame = now()

                if pcm is None:
                    stopped_tts = True
                    break

                if t_first_pcm is None:
                    t_first_pcm = t_frame
                    agg.ttff_total.add(t_first_pcm - t_user_stop)
                    if t_first_tts_text is not None:
                        agg.tts_delay.add(t_first_pcm - t_first_tts_text)

                if prev_frame_t is not None:
                    agg.frame_dt.add(t_frame - prev_frame_t)
                prev_frame_t = t_frame

                pcms.append(pcm)
        await act_event(act = 'lm', kind = 'end')

        # total LM decoding time = sum(tpots) ~= last_tok - first_tok (excluding TTFT)
        if (t_first_tok is not None) and (t_last_tok is not None) and (t_last_tok > t_first_tok):
            agg.lm_decode.add(t_last_tok - t_first_tok)

        tts_in.put_nowait("<eos>")

        # flush remaining TTS frames (also captures first_pcm if it didn't arrive during draining)
        while not stopped_tts:
            t_frame, pcm = await tts_out.get()
            # t_frame = now()

            if pcm is None:
                stopped_tts = True
                break

            if t_first_pcm is None:
                t_first_pcm = t_frame
                agg.ttff_total.add(t_first_pcm - t_user_stop)
                if t_first_tts_text is not None:
                    agg.tts_delay.add(t_first_pcm - t_first_tts_text)

            if prev_frame_t is not None:
                agg.frame_dt.add(t_frame - prev_frame_t)
            prev_frame_t = t_frame

            pcms.append(pcm)
        await act_event(act = 'tts', kind = 'end')
        audio_output = np.concatenate(pcms) if pcms else np.zeros((0,), dtype=np.float32)
        audio_outputs.append(audio_output)

        text_output = "".join(text_outputs)
        messages.append({"role": "assistant", "content": text_output})

        # -----------------------
        # tqdm one-line logging
        # -----------------------
        tqdm_construct.set_postfix_str(
            f"turn {rid//2} | {agg.one_liner()}",
            refresh=False
        )

    return (
        req.req_id,
        messages,
        audio_outputs,
        agg,
    )


def merge_aggs(aggs: List[ChatTimingAgg]) -> ChatTimingAgg:
    out = ChatTimingAgg()
    for a in aggs:
        for k, v in out.__dict__.items():
            getattr(out, k).xs.extend(getattr(a, k).xs)
    return out
def flatten_summary(summary: dict) -> dict:
    flat = {}
    for metric_name, stats in summary.items():
        for stat_name, v in stats.items():
            flat[(metric_name, stat_name)] = v
    return flat

@dataclass 
class Bench:
    arrival_rate: float = 1.0
    arrival_pattern: str = 'poisson'
    dataset_name: str  = 'MultiD'
    output_dir: Path = 'outputs'
    requests: list[Request] = field(init = False)
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.dataset_name == "MultiD":
            ds = load_dataset("IVLLab/MultiDialog", "valid_freq")['valid_freq']
            # print(ds)
            # exit(0)
            tokenizer = tiktoken.encoding_for_model('gpt-5')
            i = 0
            requests = []
            while i < len(ds):
                conv_id = ds[i]['conv_id']
                role = 'human'
                scripts = []
                while i < len(ds) and ds[i]['conv_id'] == conv_id:
                    if ds[i]['from'] != role: 
                        i += 1
                        continue
                    
                    scripts.append(Script(
                        audio = resample_poly(ds[i]['audio']['array'], up = 3, down = 2),
                        n_tokens=len(tokenizer.encode(ds[i]['value'])),
                        text = ds[i]['value'],
                        speaker = role
                    ))
                    
                    if role == 'human': role = 'gpt'
                    elif role == 'gpt': role = 'human'
                    i += 1
                scripts = scripts[:len(scripts) // 2 * 2]
                if len(scripts):
                    requests.append(Request(
                        f'req-{i}-{conv_id}',
                        scripts
                    ))
                    
        else: 
            raise NotImplementedError(f'Unkown dataset {self.dataset_name}')
        
        self.requests = requests
        script_lens = list(map(lambda r: len(r.scripts), self.requests))
        n_tokens = list(map(lambda r: np.mean([s.n_tokens for s in r.scripts]), self.requests))
        logger.info(f'collected {len(self.requests)} requests, #turns {np.mean(script_lens)}+-{np.std(script_lens)}, #tokens {np.mean(n_tokens)}+-{np.std(n_tokens)}')
    
    
    
    async def run(self, first_n:int = None):
        start_t = time.time()
        
        logger.info(f"Starting up takes {time.time() - start_t} seconds")
        # try:
        if first_n is None: 
            first_n = len(self.requests)
        assert self.arrival_pattern == 'poisson', self.arrival_pattern
        
        async with UsageMonitor(
            interval = 0.05,
            samplers = {
                'stt_mem': stt_engine.get_mem_usage, 
                'tts_mem': tts_engine.get_mem_usage, 
                'lm_mem': make_kv_cache_sampler(),
                'sm': make_nvml_sm_sampler(),
                'inflight_stt': lambda: act_counts['stt'],
                'inflight_tts': lambda: act_counts['tts'],
                'inflight_lm': lambda: act_counts['lm'],
                'inflight_tot': lambda: act_counts['stt'] + act_counts['tts'] + act_counts['lm']
            }
            ) as mon:
            tasks = []
            rng = np.random.default_rng()
            for req in tqdm.tqdm(self.requests[:first_n], desc = 'sending request'):
                tasks.append(asyncio.create_task(
                    process_request(req)
                ))
                gap = rng.exponential(1 / self.arrival_rate)
                await asyncio.sleep(gap)

            results = await tqdm_asyncio.gather(*tasks)
            
        self.output_dir.mkdir(exist_ok=True)
        project_dir = self.output_dir / f"{self.dataset_name}-{self.arrival_pattern}-{self.arrival_rate}-{first_n}"
        project_dir.mkdir(exist_ok=True)
        (project_dir / "usage.json").write_text(json.dumps(mon.results, indent = 2))
        
        # store per-request metrics JSON (already)
        all_messages = {}
        all_metrics = {}
        for req_id, messages, audios, agg in results:
            all_messages[req_id] = messages
            all_metrics[req_id] = agg.summary()  # <-- store summary dict, JSON-serializable

        (project_dir / "messages.json").write_text(json.dumps(all_messages, indent=2))
        (project_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

        # ---- run-level aggregation (merge aggs from results) ----
        run_agg = merge_aggs([agg for _, _, _, agg in results])  # if results returns agg (ChatTimingAgg)
        run_summary = run_agg.summary()                           # dict: metric -> stats
        flat = flatten_summary(run_summary)                       # dict: (metric, stat) -> value

        # add metadata rows (optional but very handy)
        flat[("meta", "dataset")] = self.dataset_name
        flat[("meta", "arrival")] = self.arrival_pattern
        flat[("meta", "arrival_rate")] = float(self.arrival_rate)
        flat[("meta", "first_n")] = int(first_n)

        run_id = f"{self.dataset_name}-{self.arrival_pattern}-{self.arrival_rate}-{first_n}"
        col = pd.Series(flat, name=run_id)

        result_path = self.output_dir / "result.csv"

        if result_path.exists():
            df = pd.read_csv(result_path, header=0, index_col=[0, 1])
            # ensure MultiIndex is exactly two levels
            df.index = pd.MultiIndex.from_tuples(df.index, names=["metric", "stat"])
            df[run_id] = col
        else:
            df = col.to_frame()

        # enforce MultiIndex names even on first write
        df.index = pd.MultiIndex.from_tuples(df.index, names=["metric", "stat"])
        df.to_csv(result_path)

        logger.info(f"result saved to {result_path}")
        

async def main(args):
    global batch_size 
    batch_size = args.batch_size
    await _init()
    for first_n in tqdm.tqdm(args.first_n, desc = 'benchmark'):
        await bench.run(first_n = first_n)
    for task in (stt_engine_task, tts_engine_task):
        if task is not None:
            task.cancel()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--first_n', type = int, nargs = '+', default = [1])
parser.add_argument('--batch_size', type = int, default = 16)
args = parser.parse_args()

bench = Bench()

asyncio.run(main(args))
