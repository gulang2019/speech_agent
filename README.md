# LM Performance Test Framework

测量 transformer-based 模型在不同到达率 (lambda) 下的 P99 latency。

## 支持的模型类型

- **ASR**: Whisper (openai/whisper-tiny, openai/whisper-base)
- **LM**: LLaMA, Qwen, TinyLlama
- **Seq2Seq**: T5, Flan-T5

## 项目结构

```
LM_perf_test/
├── config/
│   └── models.yaml          # 模型配置列表
├── src/
│   ├── model_loader.py      # 模型加载抽象
│   ├── models/
│   │   ├── asr.py          # Whisper 模型
│   │   └── lm.py            # LLM / Seq2Seq 模型
│   ├── workload.py          # Poisson 到达过程
│   ├── executor.py          # Batch 推理执行器
│   ├── metrics.py           # P99 计算
│   └── analyzer.py          # CSV + Plot 输出
├── scripts/
│   └── run_benchmark.py     # 主入口
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 测试所有模型

```bash
python scripts/run_benchmark.py --lambdas 1 5 10 50 100
```

### 测试单个模型

```bash
python scripts/run_benchmark.py --model tinyllama --lambdas 1 5 10
```

### 指定测试时长

```bash
python scripts/run_benchmark.py --model whisper-tiny --duration 60
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 指定模型名称（不指定则测试所有模型） | None |
| `--lambdas` | 到达率列表 (jobs/sec) | [1, 5, 10, 50] |
| `--duration` | 每个 lambda 的测试持续时间 (秒) | 30.0 |
| `--config` | 模型配置文件路径 | config/models.yaml |
| `--output-dir` | 输出目录 | outputs |

## 添加自定义模型

编辑 `config/models.yaml`:

```yaml
models:
  - name: my-model
    type: lm           # lm, asr, seq2seq
    path: path/to/model
    max_tokens: 256    # LM/Seq2Seq 用
    batch_size: 8
    # asr 模型用:
    # audio_length_ms: 30000
    # batch_size: 4
```

## 输出

- **CSV**: `outputs/<model_name>_results.csv`
  - lambda: 到达率
  - p99_latency: P99 延迟（秒）
  - throughput: 吞吐率（jobs/sec）
  - num_requests: 请求总数

- **对比图**: `outputs/all_models_comparison.png`
  - P99 latency vs lambda 曲线对比

## 工作原理

1. Poisson 过程生成请求，到达间隔 ~ Exponential(1/lambda)
2. 请求加入队列，BatchExecutor 按 batch_size 收集后批量推理
3. 记录每个请求延迟，计算 P99
4. 结果保存为 CSV 并绘图