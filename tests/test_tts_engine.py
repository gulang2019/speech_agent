from speech_agent.async_tts_engine import TTSEngine
import wave
import asyncio
import numpy as np

texts = ["Moshi models two streams of audio: one corresponds to Moshi speaking, and the other one to the user speaking. Along with these two audio streams, Moshi predicts text tokens corresponding to its own speech, its inner monologue, which greatly improves the quality of its generation. A small Depth Transformer models inter-codebook dependencies for a given time step, while a large, 7B-parameter Temporal Transformer models the temporal dependencies. Moshi achieves a theoretical latency of 160ms (80ms for the frame size of Mimi + 80ms of acoustic delay), with a practical overall latency as low as 200ms on an L4 GPU.",
         "Fuck It... I cannot find my passport :)",
         "I love you girl! could you marry me?", 
         "Just want to make it clear, no pain, no gain,"]
voice = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
# voice = "abstract.mp3"
engine = TTSEngine(voice = voice)


def test():
    # engine_task = asyncio.create_task(engine.step())
    # engine_task.add_done_callback(on_done)
    
    in_q, out_q = engine.add_request('req-1')

    for word in texts[0].split():
        in_q.put_nowait(word)
    in_q.put_nowait("<eos>")
    
    i = 0
    while engine.on:
        i += 1
        print('step', i)
        engine._step()
    
    pcms = []
    while not out_q.empty():        
        pcm = out_q.get_nowait()
        if pcm is None: 
            break 
        pcms.append(pcm)
    
    audio = np.concatenate(pcms, axis=-1)
    audio_f32 = audio.astype(np.float32)
    audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)


    out_path = "tts_out.wav"
    sr = int(engine.sample_rate)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(2)      # int16
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())

    print(f"Wrote {out_path} (sr={sr}, samples={audio_i16.shape[0]})")

def test_async():
    def on_done(t: asyncio.Task):
        try:
            t.result()   # re-raises exception if task failed
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print("Background task failed", e)

    async def run_request(text):
        tag = text[:10].replace(' ', '_')
        in_q, out_q = engine.add_request(f'req-{tag}')
        pcms = []
        stopped = False 
        for word in text.split() + ['<eos>']:
            in_q.put_nowait(word)
            while not out_q.empty():
                pcm = out_q.get_nowait()
                if pcm is None: 
                    stopped = True 
                    break 
                pcms.append(pcm)
            await asyncio.sleep(0.05)
    
        while not stopped:
            pcm = await out_q.get()
            if pcm is None: 
                stopped = True 
                break 
            pcms.append(pcm)
            
        audio = np.concatenate(pcms, axis=-1)
        audio_f32 = audio.astype(np.float32)
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)


        out_path = f"outputs/tts-{tag}.wav"
        sr = int(engine.sample_rate)

        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)      # mono
            wf.setsampwidth(2)      # int16
            wf.setframerate(sr)
            wf.writeframes(audio_i16.tobytes())

        print(f"Wrote {out_path} (sr={sr}, samples={audio_i16.shape[0]})")

    async def main():
        engine_task = asyncio.create_task(engine.step())
        engine_task.add_done_callback(on_done)

        tasks = []
        for text in texts:
            task = asyncio.create_task(run_request(text))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
    asyncio.run(main())
test()
# test_async()