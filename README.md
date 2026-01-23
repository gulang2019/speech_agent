# speech_agent

```
git submodule update --init
```

## Run mini-vllm
```bash
cd 3rdparty/vllm
pip install -e .
```

Run Offline inference.
```bash
export PYTHONPATH=$PWD 
python tests/mini_vllm/test_offline.py
```

Run Online inference Serving
in the first terminal, 
```bash
python -m mini_vllm.api_server --model_name facebook/opt-125m --max_memory_utilization 0.8 
```
in the next terminal
```sh
curl http://localhost:8000/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":64,"ignore_eos":false}'```
```

## Run speech agent 

Setup the environment
```bash
pip install moshi
```

In the first terminal 
```bash
vllm serve google/gemma-3-4b-it --gpu_memory_utilization 0.40 --api-key tok-123
```

In the second terminal, run
```bash
VLLM_MODEL=google/gemma-3-4b-it python benchmark/benchmark.py --first_n 1 4 8 16 32 48 64 128 --batch_size 16
```