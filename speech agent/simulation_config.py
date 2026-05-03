"""Configuration constants and utilities for the simulation."""

import json
import os

# Default SLO thresholds
DEFAULT_SLO = {"TTFF": 5.0, "TBF": 0.02}

# Generation ratios when predestined_gen_len is not provided
ST_RATIO = 0.25   # STT: output text tokens ≈ input frames * ST_ratio
TS_RATIO = 4      # TTS: output audio frames ≈ input text tokens * TS_ratio
MAX_SEQ_LEN = 50  # LM fallback: max generation length
EOS_PROB = 0.05   # LM fallback: EOS probability (geometric distribution)

# Average conversation rounds per user (used for round_arrival_rate)
AVG_ROUNDS_PER_CONV = 10.51


def load_perf_config(config_path=None):
    """Load performance configuration from JSON file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'mini_vllm', 'perf_data.json')

    with open(config_path, "r") as f:
        return json.load(f)


def get_model_perf_config(model_name, config_path=None):
    """Get performance config for a specific model."""
    config_data = load_perf_config(config_path)
    if model_name in config_data:
        return config_data[model_name]
    return config_data.get("default", {})


def compute_model_time(batch_size, query_lens, context_lens, perf_config, sm_factor=1.0):
    """
    Compute prefill/decode time for a batch.

    Args:
        batch_size: number of requests
        query_lens: list of query lengths
        context_lens: list of context lengths
        perf_config: performance config dict with 'prefill' and 'decode' coefficients
        sm_factor: SM shortage factor (1.0 = no shortage)

    Returns:
        (prefill_time_ms, decode_time_ms)
    """
    prefill_cfg = perf_config.get("prefill", [2e-6, 4e-2, 7.0])
    decode_cfg = perf_config.get("decode", [7e-5, 4e-1, 0.013])

    B = batch_size
    Q_max = max(query_lens) if query_lens else 1
    C_max = max(context_lens) if context_lens else 1

    prefill_time = (prefill_cfg[0] * (B * Q_max ** 2)
                   + prefill_cfg[1] * (B * Q_max)
                   + prefill_cfg[2]) if prefill_cfg else 0.001
    decode_time = (decode_cfg[0] * (B * C_max)
                  + decode_cfg[1] * B
                  + decode_cfg[2]) if decode_cfg else 0.001

    prefill_time *= sm_factor
    decode_time *= sm_factor

    return max(0.001, prefill_time), max(0.001, decode_time)


def parse_ec_config(ec_config_str, wq_config_str=None, num_models=3):
    """
    Parse EC configuration from CLI arguments.

    Args:
        ec_config_str: comma-separated SM allocation ratios (e.g. '10,60,30')
        wq_config_str: comma-separated WQ concurrency limits (e.g. '8,4,4')
        num_models: number of models in pipeline

    Returns:
        list of GreenContextConfig objects, or None if not provided
    """
    if ec_config_str is None:
        return None

    from request_utils import GreenContextConfig, SMResource, WQConfig

    sm_ratios = [float(x) / 100.0 for x in ec_config_str.split(',')]
    wq_limits = [int(x) for x in wq_config_str.split(',')] if wq_config_str else [8] * len(sm_ratios)

    if len(sm_ratios) != len(wq_limits):
        raise ValueError("--ec-config and --wq-config must have the same number of elements")

    ec_configs = []
    for i, (sm_ratio, wq_limit) in enumerate(zip(sm_ratios, wq_limits)):
        ec_configs.append(GreenContextConfig(
            model_index=i,
            sm_resource=SMResource(),
            wq_config=WQConfig(wq_concurrency_limit=wq_limit),
            sm_allocation_ratio=sm_ratio,
        ))

    return ec_configs