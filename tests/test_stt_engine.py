"""
Smoke test for your Engine.

What it checks (minimal end-to-end):
1) Can create Engine
2) Can add a request and get (input_queue, output_queue)
3) Can push a few PCM chunks, then '<eos>'
4) Engine emits some text tokens (maybe blanks/spaces) and eventually '<eoa>'
5) Engine returns the slot to the free pool (implicitly, by not hanging)

Run:
  python smoke_test_engine.py

Notes:
- This assumes your Engine class is in the same file, or importable.
- If the real model is heavy / slow on first run, the test timeouts may need adjustment.
"""

import asyncio
import time
import torch
import sphn

# If Engine is in another module, import it:
from speech_agent.async_stt_engine import Engine


async def _drain_until_eoa(out_q: asyncio.Queue, timeout_s: float = 20.0):
    """Collect outputs until '<eoa>' or timeout."""
    outputs = []
    t0 = time.time()
    while True:
        remaining = timeout_s - (time.time() - t0)
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for <eoa>. Collected: {outputs[:20]} ...")

        msg = await asyncio.wait_for(out_q.get(), timeout=remaining)
        outputs.append(msg)
        if msg == "<eoa>":
            return outputs


async def smoke_test_single_request():
    # Use smaller batch to make the test lighter
    engine = Engine(model_name="kyutai/stt-2.6b-en", device="cuda", batch_size=4)

    # Start engine loop
    engine_task = asyncio.create_task(engine.step())

    # Add one request
    in_q, out_q = await engine.add_request(req_id="req-smoke-1")

    frame_size = engine.frame_size

    # Create a few "silence" frames on CPU; engine will move to GPU if you do .to(device) inside prepare_input,
    # otherwise push them already on GPU:
    # frame = torch.zeros((1, 1, frame_size), dtype=torch.float32, device=engine.device)
    frame = torch.zeros((1, 1, frame_size), dtype=torch.float32)  # CPU is okay if your engine handles it

    # Push N frames (simulate short utterance / silence)
    N = 20
    for _ in range(N):
        in_q.put_nowait(frame)

    # Signal end-of-speech (your engine will append right-pad frames and then <eoa>)
    in_q.put_nowait("<eos>")

    # Wait for outputs until <eoa>
    outputs = await _drain_until_eoa(out_q, timeout_s=30.0)

    # Stop the engine task
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        pass

    # Basic assertions (don’t assume tokens are meaningful with silence input)
    assert outputs[-1] == "<eoa>", f"Last output was not <eoa>: {outputs[-5:]}"
    # It’s okay if the model emits nothing before <eoa>, but usually you’ll see some tokens.
    print("PASS: got <eoa>")
    print("Sample outputs:", outputs[:30], "...")


async def smoke_test_two_concurrent_requests():
    engine = Engine(model_name="kyutai/stt-2.6b-en", device="cuda", batch_size=4)
    engine_task = asyncio.create_task(engine.step())

    in_q1, out_q1 = await engine.add_request(req_id="req-1")
    in_q2, out_q2 = await engine.add_request(req_id="req-2")

    frame_size = engine.frame_size
    frame = torch.zeros((1, 1, frame_size), dtype=torch.float32)

    # Request 1: shorter
    for _ in range(10):
        in_q1.put_nowait(frame)
    in_q1.put_nowait("<eos>")

    # Request 2: longer
    for _ in range(30):
        in_q2.put_nowait(frame)
    in_q2.put_nowait("<eos>")

    # Collect both
    outs1_task = asyncio.create_task(_drain_until_eoa(out_q1, timeout_s=30.0))
    outs2_task = asyncio.create_task(_drain_until_eoa(out_q2, timeout_s=30.0))
    outs1, outs2 = await asyncio.gather(outs1_task, outs2_task)

    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        pass

    assert outs1[-1] == "<eoa>"
    assert outs2[-1] == "<eoa>"

    print("PASS: two concurrent requests both reached <eoa>")
    print("req-1 sample:", outs1[:20], "...")
    print("req-2 sample:", outs2[:20], "...")

def on_done(t: asyncio.Task):
    try:
        t.result()   # re-raises exception if task failed
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print("Background task failed", e)

async def smoke_test_request():
    engine = Engine(model_name="kyutai/stt-2.6b-en", device="cuda", batch_size=4)
    engine_task = asyncio.create_task(engine.step())
    engine_task.add_done_callback(on_done)
    in_pcms, _ = sphn.read("abstract.mp3", sample_rate = engine.mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms)
    # [B, 1, h]
    in_pcms = in_pcms[None, 0:1].expand(1, -1, -1)
    chunks = [
        c
        for c in in_pcms.split(engine.frame_size, dim = 2)
        if c.shape[-1] == engine.frame_size
    ]

    in_q1, out_q1 = await engine.add_request(req_id="req-1")
    for c in chunks:
        in_q1.put_nowait(c)
    in_q1.put_nowait('<eos>')
    while True:
        t = await out_q1.get()
        print(t, end = '', flush = True)
        if t == '<eoa>': break 
    engine_task.cancel()
if __name__ == "__main__":
    # Choose one (or run both)
    # asyncio.run(smoke_test_single_request())
    # asyncio.run(smoke_test_two_concurrent_requests())
    asyncio.run(smoke_test_request())
