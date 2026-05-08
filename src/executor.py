import time
import threading
from typing import List, Callable, Any
from dataclasses import dataclass
from .workload import Job


@dataclass
class ExecutionResult:
    job_id: int
    latency: float
    success: bool
    error: str = None


class BatchExecutor:
    def __init__(self, model: Any, batch_size: int, max_queue_size: int = 1000):
        self.model = model
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self._results = []
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread = None

    def submit(self, job: Job):
        self._job_queue.append(job)

    def start(self):
        self._running = True
        self._job_queue = []
        self._worker_thread = threading.Thread(target=self._process_loop)
        self._worker_thread.start()

    def stop(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join()

    def _process_loop(self):
        while self._running:
            if len(self._job_queue) >= self.batch_size:
                batch = self._job_queue[:self.batch_size]
                self._job_queue = self._job_queue[self.batch_size:]
                self._execute_batch(batch)
            else:
                time.sleep(0.001)

    def _execute_batch(self, batch: List[Job]):
        start_time = time.time()
        try:
            if hasattr(self.model, 'generate_batch'):
                payloads = [job.payload for job in batch]
                self.model.generate_batch(payloads)
            end_time = time.time()
            batch_time = end_time - start_time

            with self._lock:
                for i, job in enumerate(batch):
                    result = ExecutionResult(
                        job_id=job.job_id,
                        latency=batch_time / len(batch),
                        success=True
                    )
                    self._results.append(result)
        except Exception as e:
            with self._lock:
                for job in batch:
                    result = ExecutionResult(
                        job_id=job.job_id,
                        latency=time.time() - start_time,
                        success=False,
                        error=str(e)
                    )
                    self._results.append(result)

    def get_results(self) -> List[ExecutionResult]:
        with self._lock:
            return self._results.copy()