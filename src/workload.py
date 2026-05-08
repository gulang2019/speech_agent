import numpy as np
import queue
import threading
import time
from dataclasses import dataclass
from typing import List


@dataclass
class Job:
    job_id: int
    arrival_time: float
    payload: any


class PoissonWorkloadGenerator:
    def __init__(self, lambda_rate: float, job_size: int, job_type: str = 'tokens'):
        self.lambda_rate = lambda_rate
        self.job_size = job_size
        self.job_type = job_type
        self.job_counter = 0
        self._running = False
        self._thread = None
        self._job_queue = queue.Queue()

    def _generate_jobs(self):
        interval = 1.0 / self.lambda_rate
        while self._running:
            self.job_counter += 1
            job = Job(
                job_id=self.job_counter,
                arrival_time=time.time(),
                payload=self._generate_payload()
            )
            self._job_queue.put(job)
            interval = np.random.exponential(1.0 / self.lambda_rate)
            time.sleep(interval)

    def _generate_payload(self):
        if self.job_type == 'audio':
            sample_rate = 16000
            duration_ms = self.job_size
            num_samples = int(sample_rate * duration_ms / 1000)
            return np.random.randn(num_samples).astype(np.float32)
        else:
            num_tokens = self.job_size
            return f"Generate a response of approximately {num_tokens} tokens."

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._generate_jobs)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def get_job_queue(self) -> queue.Queue:
        return self._job_queue


class WorkloadManager:
    def __init__(self, lambdas: List[float], job_size: int, job_type: str):
        self.lambdas = lambdas
        self.job_size = job_size
        self.job_type = job_type
        self.generators = []

    def create_generator(self, lambda_rate: float) -> PoissonWorkloadGenerator:
        gen = PoissonWorkloadGenerator(lambda_rate, self.job_size, self.job_type)
        self.generators.append(gen)
        return gen