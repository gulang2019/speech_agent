from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import csv
import json
import os
import sys
import time
from typing import Iterable, Optional

import torch

from mini_vllm.profile.batch_sampler import BatchSpec
from mini_vllm.profile.energy_meter import EnergyMeter
from mini_vllm.struct import Batch


class _ResultStore:
    def __init__(self, path: Optional[str], resume: bool) -> None:
        self.path = path
        self.rows: list[dict[str, object]] = []
        if path is None:
            return
        if resume and os.path.exists(path):
            self.rows = self._load(path)
        else:
            with open(path, "w", encoding="utf-8", newline="") as f:
                if path.endswith(".csv"):
                    f.write("")

    def append(self, row: dict[str, object]) -> None:
        row_copy = dict(row)
        self.rows.append(row_copy)
        if self.path is None:
            return
        if self.path.endswith(".csv"):
            self._write_csv()
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row_copy) + "\n")

    def _write_csv(self) -> None:
        if self.path is None:
            return
        if not self.rows:
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                f.write("")
            return
        keys = sorted({k for row in self.rows for k in row.keys()})
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def _load(self, path: str) -> list[dict[str, object]]:
        if path.endswith(".csv"):
            with open(path, "r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        rows: list[dict[str, object]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows


class _ResumeState:
    def __init__(self, rows: Iterable[dict[str, object]], skip_failed: bool) -> None:
        self.idle_recorded = False
        self.completed_repeats: dict[str, set[int]] = defaultdict(set)
        self.failed_batches: set[str] = set()

        for row in rows:
            record_type = row.get("record_type")
            if record_type == "idle":
                self.idle_recorded = True
                continue
            batch_key = self._row_batch_key(row)
            if batch_key is None:
                continue
            if record_type == "batch":
                repeat = self._to_int(row.get("repeat"))
                if repeat is not None:
                    self.completed_repeats[batch_key].add(repeat)
            elif record_type == "error" and skip_failed:
                self.failed_batches.add(batch_key)

    def pending_repeats(self, batch_key: str, repeats: int) -> list[int]:
        completed = self.completed_repeats.get(batch_key, set())
        return [rep for rep in range(repeats) if rep not in completed]

    def should_skip_batch(
        self,
        batch_key: str,
        repeats: int,
        retry_failed: bool,
    ) -> bool:
        if not retry_failed and batch_key in self.failed_batches:
            return True
        return len(self.completed_repeats.get(batch_key, set())) >= repeats

    def mark_repeat_completed(self, batch_key: str, repeat: int) -> None:
        self.completed_repeats[batch_key].add(repeat)

    def mark_batch_failed(self, batch_key: str) -> None:
        self.failed_batches.add(batch_key)

    def _row_batch_key(self, row: dict[str, object]) -> Optional[str]:
        batch_name = row.get("batch_name")
        if batch_name not in (None, ""):
            return f"name:{batch_name}"
        batch_index = self._to_int(row.get("batch_index"))
        if batch_index is None:
            return None
        return f"index:{batch_index}"

    def _to_int(self, value: object) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


class BatchProfiler:
    def __init__(
        self,
        model_runner,
        energy_meter: EnergyMeter,
        warmup: int = 2,
        warmup_s: float = 10.0,
        repeats: int = 20,
        warmup_scope: str = "global",
        sync_cuda: bool = True,
        show_progress: bool = True,
    ):
        self.model_runner = model_runner
        self.energy_meter = energy_meter
        self.warmup = warmup
        self.warmup_s = warmup_s
        self.repeats = repeats
        if warmup_scope not in {"global", "batch"}:
            raise ValueError("warmup_scope must be 'global' or 'batch'")
        self.warmup_scope = warmup_scope
        self.sync_cuda = sync_cuda
        self.show_progress = show_progress

    def profile(
        self,
        batches: Optional[Iterable[Batch]] = None,
        specs: Optional[Iterable[BatchSpec]] = None,
        batch_builder: Optional[Callable[[BatchSpec], Batch]] = None,
        output_path: Optional[str] = None,
        idle_s: float = 2.0,
        resume: bool = False,
        retry_failed: bool = False,
        continue_on_error: bool = False,
    ) -> list[dict[str, object]]:
        specs_list = list(specs) if specs is not None else []
        store = _ResultStore(output_path, resume=resume)
        results = store.rows
        resume_state = _ResumeState(results, skip_failed=resume)
        global_warmup_done = self.warmup_scope != "global" or not self._has_warmup()

        if idle_s > 0 and not resume_state.idle_recorded:
            idle_stats = self._measure_idle(idle_s)
            store.append({"record_type": "idle", **idle_stats})

        if batch_builder is not None:
            outer = self._wrap_progress(list(enumerate(specs_list)), desc="Batches")
            for idx, spec in outer:
                warmup_completed = self._profile_spec(
                    batch_index=idx,
                    spec=spec,
                    batch_builder=batch_builder,
                    store=store,
                    resume_state=resume_state,
                    retry_failed=retry_failed,
                    continue_on_error=continue_on_error,
                    run_warmup=not global_warmup_done,
                )
                if warmup_completed:
                    global_warmup_done = True
            return list(results)

        batches_list = list(batches) if batches is not None else []
        outer = self._wrap_progress(list(enumerate(batches_list)), desc="Batches")
        for idx, batch in outer:
            spec = specs_list[idx] if idx < len(specs_list) else None
            warmup_completed = self._profile_batch(
                batch=batch,
                batch_index=idx,
                spec=spec,
                store=store,
                resume_state=resume_state,
                retry_failed=retry_failed,
                continue_on_error=continue_on_error,
                run_warmup=not global_warmup_done,
            )
            if warmup_completed:
                global_warmup_done = True
        return results

    def _profile_spec(
        self,
        batch_index: int,
        spec: BatchSpec,
        batch_builder: Callable[[BatchSpec], Batch],
        store: _ResultStore,
        resume_state: _ResumeState,
        retry_failed: bool,
        continue_on_error: bool,
        run_warmup: bool,
    ) -> bool:
        batch_key = self._resume_batch_key(batch_index, spec)
        if resume_state.should_skip_batch(batch_key, self.repeats, retry_failed):
            return False
        try:
            batch = batch_builder(spec)
        except Exception as exc:
            if self._record_error(
                exc=exc,
                batch=None,
                batch_index=batch_index,
                spec=spec,
                phase="build",
                repeat=None,
                store=store,
                resume_state=resume_state,
                continue_on_error=continue_on_error,
            ):
                return False
            raise

        return self._profile_batch(
            batch=batch,
            batch_index=batch_index,
            spec=spec,
            store=store,
            resume_state=resume_state,
            retry_failed=retry_failed,
            continue_on_error=continue_on_error,
            run_warmup=run_warmup,
        )

    def _profile_batch(
        self,
        batch: Batch,
        batch_index: int,
        spec: Optional[BatchSpec],
        store: _ResultStore,
        resume_state: _ResumeState,
        retry_failed: bool,
        continue_on_error: bool,
        run_warmup: bool,
    ) -> bool:
        batch_key = self._resume_batch_key(batch_index, spec)
        if resume_state.should_skip_batch(batch_key, self.repeats, retry_failed):
            return False

        pending_repeats = resume_state.pending_repeats(batch_key, self.repeats)
        if not pending_repeats:
            return False

        warmup_completed = False
        if self.warmup_scope == "batch":
            run_warmup = self._has_warmup()

        if run_warmup:
            try:
                self._warmup(batch)
                warmup_completed = True
            except Exception as exc:
                if self._record_error(
                    exc=exc,
                    batch=batch,
                    batch_index=batch_index,
                    spec=spec,
                    phase="warmup",
                    repeat=None,
                    store=store,
                    resume_state=resume_state,
                    continue_on_error=continue_on_error,
                ):
                    return False
                raise

        inner = self._wrap_progress(
            pending_repeats,
            desc=f"Repeats batch {batch_index + 1}",
        )
        for rep in inner:
            try:
                record = self._measure_batch(batch)
            except Exception as exc:
                if self._record_error(
                    exc=exc,
                    batch=batch,
                    batch_index=batch_index,
                    spec=spec,
                    phase="measure",
                    repeat=rep,
                    store=store,
                    resume_state=resume_state,
                    continue_on_error=continue_on_error,
                ):
                    return
                raise

            record.update(self._batch_metadata(batch, batch_index, spec, rep))
            store.append(record)
            resume_state.mark_repeat_completed(batch_key, rep)
        return warmup_completed

    def _warmup(self, batch: Batch) -> None:
        deadline = None
        if self.warmup_s > 0:
            deadline = time.perf_counter() + self.warmup_s

        warmup_iters = 0
        while warmup_iters < self.warmup or (
            deadline is not None and time.perf_counter() < deadline
        ):
            self.model_runner.execute_batch(batch)
            warmup_iters += 1
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _has_warmup(self) -> bool:
        return self.warmup > 0 or self.warmup_s > 0

    def _measure_batch(self, batch: Batch) -> dict[str, object]:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.energy_meter.start()
        t0 = time.perf_counter()
        try:
            _ = self.model_runner.execute_batch(batch)
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        finally:
            samples = self.energy_meter.stop()
        stats = self.energy_meter.summarize(samples)

        latency_ms = (t1 - t0) * 1000.0
        record = {
            "latency_ms": latency_ms,
            "avg_power_w": stats["avg_power_w"],
            "energy_j": stats["energy_j"],
        }
        if samples:
            record["graphics_clock_mhz"] = samples[-1].graphics_clock_mhz
            record["mem_clock_mhz"] = samples[-1].mem_clock_mhz
        return record

    def _measure_idle(self, duration_s: float) -> dict[str, object]:
        samples = self.energy_meter.measure(duration_s)
        stats = self.energy_meter.summarize(samples)
        record = {
            "idle_duration_s": duration_s,
            "idle_avg_power_w": stats["avg_power_w"],
            "idle_energy_j": stats["energy_j"],
        }
        if samples:
            record["idle_graphics_clock_mhz"] = samples[-1].graphics_clock_mhz
            record["idle_mem_clock_mhz"] = samples[-1].mem_clock_mhz
        return record

    def _batch_metadata(
        self,
        batch: Batch,
        batch_index: int,
        spec: Optional[BatchSpec],
        repeat: int,
    ) -> dict[str, object]:
        total_context_tokens = sum(batch.context_lens)
        total_query_tokens = sum(batch.query_lens)
        sum_total_len_times_past_len = sum(
            (cl + ql) * cl for cl, ql in zip(batch.context_lens, batch.query_lens)
        )
        record = {
            "record_type": "batch",
            "repeat": repeat,
            "batch_index": batch_index,
            "num_reqs": len(batch.req_ids),
            "total_context_tokens": total_context_tokens,
            "total_query_tokens": total_query_tokens,
            "total_tokens": total_context_tokens + total_query_tokens,
            "total_query_len": total_query_tokens,
            "past_len": total_context_tokens,
            "sum_total_len_times_past_len": sum_total_len_times_past_len,
        }
        if spec is not None:
            record["batch_name"] = spec.name
            record["batch_type"] = spec.batch_type
        return record

    def _record_error(
        self,
        exc: Exception,
        batch: Optional[Batch],
        batch_index: int,
        spec: Optional[BatchSpec],
        phase: str,
        repeat: Optional[int],
        store: _ResultStore,
        resume_state: _ResumeState,
        continue_on_error: bool,
    ) -> bool:
        if not continue_on_error:
            return False

        record: dict[str, object] = {
            "record_type": "error",
            "batch_index": batch_index,
            "error_phase": phase,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "is_oom": self._is_oom(exc),
        }
        if repeat is not None:
            record["repeat"] = repeat
        if spec is not None:
            record["batch_name"] = spec.name
            record["batch_type"] = spec.batch_type
        if batch is not None:
            total_context_tokens = sum(batch.context_lens)
            total_query_tokens = sum(batch.query_lens)
            record.update(
                {
                    "num_reqs": len(batch.req_ids),
                    "total_context_tokens": total_context_tokens,
                    "total_query_tokens": total_query_tokens,
                    "total_tokens": total_context_tokens + total_query_tokens,
                    "total_query_len": total_query_tokens,
                    "past_len": total_context_tokens,
                }
            )

        store.append(record)
        resume_state.mark_batch_failed(self._resume_batch_key(batch_index, spec))
        self._cleanup_after_error()
        batch_label = spec.name if spec is not None else str(batch_index)
        print(
            f"Skipping batch {batch_label} after {phase} error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return True

    def _is_oom(self, exc: Exception) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        message = str(exc).lower()
        return "out of memory" in message or "cuda oom" in message

    def _cleanup_after_error(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def _resume_batch_key(self, batch_index: int, spec: Optional[BatchSpec]) -> str:
        if spec is not None:
            return f"name:{spec.name}"
        return f"index:{batch_index}"

    def _wrap_progress(self, iterable, desc: str):
        if not self.show_progress:
            return iterable
        try:
            from tqdm import tqdm  # type: ignore

            return tqdm(iterable, desc=desc)
        except Exception:
            return iterable
