import datetime
import json

from datasets import load_dataset, DownloadMode, DownloadConfig
from request_utils import BaseModel, AsyncPipelineEngine, PipelineRequest
from huggingface_hub import HfApi
from scipy.io import wavfile
from dataclasses import dataclass, field
import asyncio
import logging
import numpy as np
logger = logging.getLogger('async_pipeline_engine')

import time
import os
import random
api = HfApi(endpoint="https://hf-mirror.com")

from datasets import load_from_disk
from visual import plot_time_distribution, plot_frame_times, combine_figures_vertical
batch_sizes = [8]
lambda_rates = [200]
experiment_duration = 60.0  # seconds

final_result = {}

@dataclass
class VoiceMetadata:
    conv_id:str
    len_human_frames:int
    len_human_speech:int
    len_gpt_speech:int
    len_gpt_frames:int

    @property
    def show(self):
        return f"Conv ID: {self.conv_id}, Human Frames: {self.len_human_frames}, GPT Speech Length: {self.len_gpt_speech}, GPT Frames: {self.len_gpt_frames}\n"

@dataclass
class UserData:
    user_index:int = None
    convs:list[VoiceMetadata] = None
    conv_latencies:list[float] = field(default_factory=list)
    current_active_request:PipelineRequest = None
    current_conv_idx:int = 0


def get_wav_info(file_path,window_size = 0.025, hop_size = 0.01):
    """
    读取WAV文件，并输出其音频长度和总采样点数。

    Args:
        file_path (str): WAV文件的路径。
    """
    try:
        # 读取WAV文件
        # rate: 采样率 (samples/second)
        # data: 音频数据 (NumPy array)
        rate, data = wavfile.read(file_path)

        # 获取总采样点数
        # 如果是单声道，data.shape是一个元组 (num_samples,)
        # 如果是多声道，data.shape是一个元组 (num_samples, num_channels)
        num_samples = data.shape[0]

        # 计算音频长度（秒）
        window_size_samples = int(window_size * rate)
        hop_size_samples = int(hop_size * rate)

        frame_count = (num_samples - window_size_samples) // hop_size_samples + 1

        return frame_count, num_samples


    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"读取WAV文件时发生错误: {e}")

def prepare_data(dataset):
    convlist = []
    current_conv = []
    for i in range(len(dataset)):
        data = dataset[i]
        next_data = dataset[i + 1] if i + 1 < len(dataset) else None
        if next_data is None:
            break
        if data["from"] == "gpt" or (data["from"] == "human" and next_data and next_data["conv_id"] != data["conv_id"]) or ("valid_freq" not in data["audpath"]):
            continue
        else:
            assert data["from"] == "human", f"Expected 'human' but got {data['from']} at index {i}"
            assert next_data["from"] == "gpt", f"Expected 'gpt' but got {next_data['from']} at index {i + 1}"

            human_voice_path = os.path.join(dataset_path, data["file_name"])
            gpt_voice_path = os.path.join(dataset_path, next_data["file_name"])

            len_human_frames, human_num_samples = get_wav_info(human_voice_path)
            len_gpt_frames, gpt_num_samples = get_wav_info(gpt_voice_path)
            voice_metadata = VoiceMetadata(
                conv_id=data["conv_id"],
                len_human_speech=len(data["value"]), #TOOO: tokenizer
                len_human_frames=len_human_frames,
                len_gpt_speech=len(next_data["value"]), #TOOO: tokenizer
                len_gpt_frames=len_gpt_frames
            )
        if current_conv and data["conv_id"] != current_conv[-1].conv_id:
            convlist.append(current_conv)
            current_conv = []
            current_conv.append(voice_metadata)
        else:
            current_conv.append(voice_metadata)
    userlist = []
    for conv in convlist:
        if len(conv)>1:
            userlist.append(UserData(convs=conv))

    return userlist

async def handle_user_lifecycle(i, user, engine, wait_time, logger, stop_event=None):
    # Accumulators declared outside try so CancelledError path can still return them
    user_results = {}
    user_latencies = []
    user_ttffs = []
    user_ttnfs = []
    user_tbfs = []

    try:
        await asyncio.sleep(wait_time)

        if not user.convs:
            return None

        voice_meta = user.convs[0]
        user.convs = user.convs[1:]
        predestined_gen_length = [voice_meta.len_gpt_speech]
        human_input = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=voice_meta.len_human_frames))

        request = await engine.add_request(
            req_id=f"{i}",
            input_data=human_input,
            target_model_idx=0,
            predestined_gen_length=predestined_gen_length
        )
        user.user_index = i
        user.current_active_request = request
        logger.info(f"Added initial request for user {i}. Remaining rounds: {len(user.convs)}.\n")

        turn_count = 0
        while True:
            req = user.current_active_request
            start_wait = req.time_stamp

            timestamp, result = await req.final_output_queue.get()

            latency = timestamp - start_wait
            user_latencies.append(latency)
            user_results[f"{req.req_id}"] = (timestamp, result)
            user_ttnfs.append({
                "req_id": int(req.req_id),
                "req_input_time": req.time_stamp,
                "TTNF": req.TTNF,
            })
            ttff_val = next((t[0] for t in req.TTNF if t[1] == len(engine.model_pipeline)-1), 0) - start_wait if req.TTNF else 0
            # (arrival_time, ttff_value, completion_time) — completion_time used for active-count steady-state detection
            user_ttffs.append((req.time_stamp, ttff_val, timestamp))
            for index in range(len(req.TTNF) - 1 if req.TTNF else 1):
                if req.TTNF[index][1] == len(engine.model_pipeline)-1 and index + 1 < len(req.TTNF):
                    user_tbfs.append(req.TTNF[index + 1][0] - req.TTNF[index][0])
            logger.info(f"User {i} Turn {turn_count} finished. Latency: {latency:.4f}s")

            # Stop submitting new turns once the experiment window has closed
            if len(user.convs) == 0 or (stop_event is not None and stop_event.is_set()):
                break

            turn_count += 1
            next_meta = user.convs[0]
            user.convs = user.convs[1:]
            next_predestined = [next_meta.len_gpt_speech]
            next_input = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=next_meta.len_human_frames))

            new_request = await engine.add_request(
                req_id=f"{i}",
                input_data=next_input,
                target_model_idx=0,
                predestined_gen_length=next_predestined
            )
            user.current_active_request = new_request
            logger.info(f"Added next turn request for user {i}. Remaining rounds: {len(user.convs)}.\n")

    except asyncio.CancelledError:
        # Return whatever has been collected so far (partial data is still useful)
        pass

    return {
        "user_index": i,
        "results": user_results,
        "latencies": user_latencies,
        "TTFFs": user_ttffs,
        "TTNFs": user_ttnfs,
        "TBFs": user_tbfs
    }

def violation_calc(list,threshold):
    violation_count = sum(1 for x in list if x > threshold)
    total_count = len(list)
    violation_rate = violation_count / total_count if total_count > 0 else 0
    return violation_rate



async def single_task(userlist, lambda_rate, batch_size=4, experiment_duration=60.0, detailed_log=False, slo=None, plotting=False, ec_configs=None, dataset_path=None):


    logging.basicConfig(level=logging.INFO)
    # 定义模型工厂函数
    def create_model_1():
        return BaseModel(model_name="13B",model_index=0,model_type = "LM",device='cpu')

    model_factories = [create_model_1]

    engine = AsyncPipelineEngine(
        model_factories=model_factories,
        device='cpu',
        max_active_batch_size=batch_size,
        ec_configs=ec_configs,
    )

    engine_task = asyncio.create_task(engine.run())

    # Generate uniform arrival times within the experiment window
    # Lambda is in requests/minute, so interval = 60 / lambda_rate seconds
    if lambda_rate > 0:
        arrival_interval = 60.0 / lambda_rate
    else:
        arrival_interval = float('inf')
    arrival_times = []
    t = arrival_interval
    while t < experiment_duration:
        arrival_times.append(t)
        t += arrival_interval

    num_arrived = len(arrival_times)
    if num_arrived == 0:
        logger.warning("No users arrived within the experiment window. Try a higher lambda or longer duration.")
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass
        return

    import copy
    user_data = [copy.deepcopy(random.choice(userlist)) for _ in arrival_times]

    task_start_time = time.perf_counter()

    # stop_event fires after the arrival window; tasks finish their current turn then stop
    stop_event = asyncio.Event()
    asyncio.get_event_loop().call_later(experiment_duration, stop_event.set)

    tasks = []
    for i, (user, arrival_time) in enumerate(zip(user_data, arrival_times)):
        task = asyncio.create_task(
            handle_user_lifecycle(i, user, engine, arrival_time, logger, stop_event=stop_event)
        )
        tasks.append(task)

    # Give tasks up to arrival_window + grace to complete; force-cancel stragglers
    completion_grace = 60.0
    done, pending = await asyncio.wait(tasks, timeout=experiment_duration + completion_grace)
    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # Collect from ALL tasks (done naturally + cancelled with partial data)
    all_user_logs = []
    for task in tasks:
        try:
            result = task.result()
            if result is not None and result.get("TTFFs"):
                all_user_logs.append(result)
        except Exception:
            pass

    if len(all_user_logs) < 3:
        logger.warning(f"Only {len(all_user_logs)} users returned data. Consider lowering lambda or extending experiment_duration.")

    # --- 汇总结果 ---
    # 初始化全局汇总容器
    results = {}
    latencies = []
    TTFFs_timed = []   # list of (arrival_wallclock, ttff_value, completion_wallclock)
    TTNFs = {}
    TBFs = []
    completed_user_data = []

    # 将各用户的数据合并到全局容器中
    for log_data in all_user_logs:
        if log_data is None: continue # 跳过空数据

        results.update(log_data["results"])
        latencies.extend(log_data["latencies"])
        TTFFs_timed.extend(log_data["TTFFs"])
        TTNFs[f"{log_data['user_index']}"] = log_data["TTNFs"]
        TBFs.extend(log_data["TBFs"])

        # 如果需要保留修改后的 user 对象
        # completed_user_data.append(...)

    # --- 稳态检测 + TTFF slope ---
    TTFFs = [v for _, v, _ in TTFFs_timed]

    ttff_slope = float('nan')
    ss_start_idx = 0          # 稳态起始样本下标
    max_active = 0

    if len(TTFFs_timed) > 1:
        # 按到达时刻排序
        TTFFs_sorted = sorted(TTFFs_timed, key=lambda x: x[0])
        # 计算每个请求到达时的并发活跃数 active_count(t) = #{i : arrival_i <= t < completion_i}
        arrivals   = np.array([a for a, _, _ in TTFFs_sorted])
        completions = np.array([c for _, _, c in TTFFs_sorted])
        active_counts = np.array([
            int(np.sum((arrivals <= arr_t) & (completions > arr_t)))
            for arr_t in arrivals
        ])

        max_active = int(active_counts.max())
        # 稳态起点：active_count 首次达到峰值 80% 的位置
        ss_threshold = 0.8 * max_active
        ss_candidates = np.where(active_counts >= ss_threshold)[0]
        ss_start_idx = int(ss_candidates[0]) if len(ss_candidates) else 0

        ss_data = TTFFs_sorted[ss_start_idx:]
        if len(ss_data) > 1:
            arrival_rel_ss = np.array([t - task_start_time for t, _, _ in ss_data])
            ttff_arr_ss    = np.array([v for _, v, _ in ss_data])
            ttff_slope = float(np.polyfit(arrival_rel_ss, ttff_arr_ss, 1)[0])

    logger.info(
        f"Steady-state detection: max_active={max_active}, "
        f"ss_start_idx={ss_start_idx}/{len(TTFFs_sorted)}, "
        f"slope computed on {len(TTFFs_sorted)-ss_start_idx} samples"
    )

    # --- 打印最终统计信息 ---
    log_info = (
        f"\n\nInfo of this run:\n"
        f"- Users arrived: {num_arrived}\n"
        f"- Experiment duration: {experiment_duration:.1f}s\n"
        f"- Lambda: {lambda_rate:.2f}\n"
        f"- Batch size: {batch_size}\n\n"
    )
    detailed_info = log_info

    log_info += f"Average latency: {1000*np.mean(latencies):.4f}ms, Std latency: {np.std(latencies):.4f}, Max latency: {1000*np.max(latencies):.4f}ms, Min latency: {1000*np.min(latencies):.4f}ms\nAverage TTFF: {1000*np.mean(TTFFs):.4f}ms, average TBFs: {1000*np.mean(TBFs):.4f}ms, Average Batch Size: {np.mean(engine.batching_data):.2f}\nTTFF slope (post-warmup): {ttff_slope:.6f} s/s [warmup={ss_start_idx}/{len(TTFFs_timed)} samples, max_active={max_active}]\n"
    logger.info(f"Average latency: {1000*np.mean(latencies):.4f}ms, Std latency: {np.std(latencies):.4f}, Max latency: {1000*np.max(latencies):.4f}ms, Min latency: {1000*np.min(latencies):.4f}ms\nAverage TTFF: {1000*np.mean(TTFFs):.4f}ms, average TBFs: {1000*np.mean(TBFs):.4f}ms, Average Batch Size: {np.mean(engine.batching_data):.2f}\nTTFF slope (post-warmup): {ttff_slope:.6f} s/s [warmup={ss_start_idx}/{len(TTFFs_timed)} samples, max_active={max_active}]")
    
    for user in completed_user_data:
        log_info += f"User {user.user_index} conversation latencies: {user.conv_latencies}, Average: {np.mean(user.conv_latencies):.4f}\n"
        #logger.info(f"User {user.user_index} conversation latencies: {user.conv_latencies}, Average: {np.mean(user.conv_latencies):.4f}\n")
    
    detailed_info += f"\nDetailed per-request info:\n"
    detailed_info += f"All latencies: {latencies}\nAll TTFFs: {TTFFs}\nAll TBFs: {TBFs}\n"    
    
    with open("real_run_log.txt", "a") as f:
        f.write(log_info)
    if detailed_log:
        with open("real_run_detailed_log.txt", "a") as f:
            f.write(detailed_info)
    
    if slo is None:
        slo = {"TTFF": 5, "TBF": 0.02}

    ttff_violation_rate = violation_calc(TTFFs, slo["TTFF"])
    tbf_violation_rate = violation_calc(TBFs, slo["TBF"])

    final_result[f"batch_size_{batch_size}_lambda_{lambda_rate}"] = {
        "num_arrived": num_arrived,
        "arrival_rate": num_arrived / experiment_duration if experiment_duration > 0 else 0,
        "round_arrival_rate": num_arrived * 10.51 / experiment_duration if experiment_duration > 0 else 0,
        "experiment_duration": experiment_duration,
        "average_latency": np.mean(latencies),
        "average_TTFF": np.mean(TTFFs),
        "average_TBF": np.mean(TBFs),
        "TBF_p99": np.percentile(TBFs, 99) if TBFs else float('nan'),
        "TTFF_slope_s_per_s": ttff_slope,
        "slope_warmup_samples": ss_start_idx,
        "slope_total_samples": len(TTFFs_timed),
        "max_active_users": max_active,
        "average_batch_size": np.mean(engine.batching_data),
        "TTFF_violation_rate": ttff_violation_rate,
        "TBF_violation_rate": tbf_violation_rate,
        "throughput": {
            model.model_type: {
                "total_frames": model.total_requests_processed,
                "total_rounds": model.total_rounds_completed,
                "runtime_s": model.total_processing_time / 1000.0,
                "frame_throughput": model.total_requests_processed / (model.total_processing_time / 1000.0) if model.total_processing_time > 0 else 0,
                "round_throughput": model.total_rounds_completed / (model.total_processing_time / 1000.0) if model.total_processing_time > 0 else 0,
            }
            for model in engine.model_pipeline
        },
    }

    if plotting == True:
        plot_list = []
        plot_frame_times(task_start_time,TTNFs, title=f"Batch Size {batch_size} Lambda {lambda_rate} TTNF Distribution Experiment in time {datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')}", clipp_range_max=300)
        for i in range(len(engine.model_pipeline)):
            plot_list.append(plot_time_distribution(engine.queuing_task_counter[i], interval= engine.sampling_interval, title=f"Batch Size {batch_size} Lambda {lambda_rate} Model {i} Queue Length Distribution",xlabel="Time (s)", ylabel="Number of Queueing Tasks"))
            plot_list.append(plot_time_distribution(engine.batch_counter[i], interval= engine.sampling_interval, title=f"Batch Size {batch_size} Lambda {lambda_rate} Model {i} Batch Size Distribution",xlabel="Time (s)", ylabel="Batch Size"))
        
        combine_figures_vertical(plot_list, title=f"Batch Size {batch_size} Lambda {lambda_rate} Combined Plot Experiment in time {datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')}")

    # Allow some time for any final logging or cleanup
    await asyncio.sleep(3)

    # Cancel the engine's main task to gracefully shut down
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        logger.info("Engine task cancelled.")

    # Write results to final_results.json (overwrite mode for clean parsing)
    with open("final_results.json", "w") as f:
        json.dump(final_result, f, indent=2)

    return final_result


if __name__ == "__main__":
    print(f"Lambda rates:{lambda_rates}, batch sizes:{batch_sizes}, experiment duration:{experiment_duration}s\n")
    dataset_path = "/data2/liuxunyuan/datasets"

    MultiD = load_dataset("/data2/liuxunyuan/datasets")["validation"]

    print("""Dataset loaded successfully. Here are some details:\n""")
    print(MultiD)
    print(MultiD[0])

    convlist = prepare_data(MultiD)

    try:
        for lambda_rate in lambda_rates:
            for batch_size in batch_sizes:
                logger.info(f"Starting test with lambda rate={lambda_rate}, batch size={batch_size}.\n")
                asyncio.run(single_task(convlist, lambda_rate=lambda_rate, batch_size=batch_size, experiment_duration=experiment_duration, detailed_log=False))
                logger.info(f"Completed test with lambda rate={lambda_rate}, batch size={batch_size}.\n")

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)

    finally:
        with open("final_results.json", "a") as f:
            json.dump(final_result, f,indent=2)

    # print(f"""Total conversations: {len(convlist)}\n""First conversation length: {len(convlist[0])}""")
    # print(f"""First conversation details:\n{[v.show for v in convlist[0]]}""")



