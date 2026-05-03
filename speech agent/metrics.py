"""Metrics collection and reporting utilities."""

import logging
import numpy as np

logger = logging.getLogger('async_pipeline_engine')


def violation_calc(values, threshold):
    """Calculate SLO violation rate: fraction of values exceeding threshold."""
    violation_count = sum(1 for x in values if x > threshold)
    total_count = len(values)
    return violation_count / total_count if total_count > 0 else 0


def compute_steady_state_stats(ttffs_timed, task_start_time):
    """
    Compute steady-state detection and TTFF slope.

    Args:
        ttffs_timed: list of (arrival_wallclock, ttff_value, completion_wallclock)
        task_start_time: experiment start wallclock time

    Returns:
        dict with ss_start_idx, max_active, ttff_slope
    """
    ttff_slope = float('nan')
    ss_start_idx = 0
    max_active = 0

    if len(ttffs_timed) <= 1:
        return {'ss_start_idx': ss_start_idx, 'max_active': max_active, 'ttff_slope': ttff_slope}

    # Sort by arrival time
    ttffs_sorted = sorted(ttffs_timed, key=lambda x: x[0])
    arrivals = np.array([a for a, _, _ in ttffs_sorted])
    completions = np.array([c for _, _, c in ttffs_sorted])

    # Compute active count at each arrival: #{i : arrival_i <= t < completion_i}
    active_counts = np.array([
        int(np.sum((arrivals <= arr_t) & (completions > arr_t)))
        for arr_t in arrivals
    ])

    max_active = int(active_counts.max())
    ss_threshold = 0.8 * max_active
    ss_candidates = np.where(active_counts >= ss_threshold)[0]
    ss_start_idx = int(ss_candidates[0]) if len(ss_candidates) else 0

    ss_data = ttffs_sorted[ss_start_idx:]
    if len(ss_data) > 1:
        arrival_rel_ss = np.array([t - task_start_time for t, _, _ in ss_data])
        ttff_arr_ss = np.array([v for _, v, _ in ss_data])
        ttff_slope = float(np.polyfit(arrival_rel_ss, ttff_arr_ss, 1)[0])

    return {
        'ss_start_idx': ss_start_idx,
        'max_active': max_active,
        'ttff_slope': ttff_slope,
        'ttffs_sorted': ttffs_sorted,
    }


def format_throughput_stats(stats):
    """Format throughput stats dict for logging."""
    lines = []
    for model_type, s in stats.items():
        lines.append(
            f"  {model_type} Throughput: frames={s['total_frames']}, "
            f"rounds={s['total_rounds']}, runtime={s['runtime_s']:.2f}s, "
            f"frame_throughput={s['frame_throughput']:.2f} frames/s, "
            f"round_throughput={s['round_throughput']:.2f} rounds/s"
        )
        lines.append(
            f"  {model_type} SS Throughput: ss_frames={s['ss_total_frames']}, "
            f"ss_rounds={s['ss_total_rounds']}, ss_wallclock={s['ss_wallclock_s']:.2f}s, "
            f"ss_frame_throughput={s['ss_frame_throughput']:.2f} frames/s, "
            f"ss_round_throughput={s['ss_round_throughput']:.2f} rounds/s"
        )
    return '\n'.join(lines)