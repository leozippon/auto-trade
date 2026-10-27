import multiprocessing
import queue
import resource  # Only available on Unix systems
import sys
import threading
import time
import traceback
from typing import Any


class CodeExecutorResult:
    def __init__(self, result_queue):
        self.result_queue = result_queue
        self._fetched = False
        self._result = None

    def fetch(self) -> Any:
        if not self._fetched:
            self._result = self.result_queue.get()
            self._fetched = True
        return self._result

    def ready(self) -> bool:
        return not self.result_queue.empty()

class CodeExecutorJob:
    def __init__(self, code, args, timeout, max_memory, result_queue):
        self.code = code
        self.args = args
        self.timeout = timeout
        self.max_memory = max_memory
        self.result_queue = result_queue
        self.process = None  # Will hold the process object
        self.start_time = None  # Will record the start time

class CodeExecutor:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.jobs_queue = queue.Queue()
        self.active_jobs = []
        self.lock = threading.Lock()
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

    def push(self, code: str, args, timeout: int = 100, max_memory: int = 100 * 1024 * 1024):
        result_queue = multiprocessing.Queue()
        result = CodeExecutorResult(result_queue)
        job = CodeExecutorJob(code, args, timeout, max_memory, result_queue)
        with self.lock:
            if len(self.active_jobs) < self.num_workers:
                self._start_job(job)
            else:
                self.jobs_queue.put(job)
        return result

    def _start_job(self, job):
        job.process = multiprocessing.Process(
            target=self._run_code,
            args=(job.code, job.args, job.result_queue, job.timeout, job.max_memory)
        )
        job.process.start()
        job.start_time = time.time()
        self.active_jobs.append(job)

    def _run_code(self, code, args, result_queue, timeout, max_memory):
        # In the child process, set resource limits
        try:
            # Set CPU time limit in seconds
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            # Set maximum memory size in bytes
            resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
        except Exception as e:
            print("Failed to set resource limits:", e)
            pass
        # Capture stdout
        from io import StringIO
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        # wrap code into def main(task) function
        # and return encode_task(task)
        # adjust the indentation

        try:
            # Now execute the code
            local_vars = {}
            exec(code, {}, local_vars)
            output = redirected_output.getvalue()
            result = local_vars.get('main', lambda x: None)(args)
            result_data = {'result': result, 'output': output}
            result_queue.put(result_data)
        except Exception as e:
            output = redirected_output.getvalue()
            error_trace = traceback.format_exc()
            result_data = {'error': str(e), 'traceback': error_trace, 'output': output}
            result_queue.put(result_data)
        finally:
            sys.stdout = old_stdout


    def _monitor(self):
        while self.running:
            with self.lock:
                for job in self.active_jobs[:]:  # Copy the list to avoid modification during iteration
                    job_elapsed_time = time.time() - job.start_time
                    if not job.process.is_alive():
                        # Job finished
                        job.process.join()
                        self.active_jobs.remove(job)
                        self._start_next_job_if_available()
                    elif job_elapsed_time > job.timeout:
                        # Job exceeded timeout
                        job.process.terminate()
                        job.process.join()
                        job.result_queue.put({'error': 'Timeout'})
                        self.active_jobs.remove(job)
                        self._start_next_job_if_available()
            time.sleep(0.1)  # Sleep a bit before next check

    def _start_next_job_if_available(self):
        if not self.jobs_queue.empty():
            next_job = self.jobs_queue.get()
            self._start_job(next_job)

    def wait(self):
        while True:
            with self.lock:
                if not self.active_jobs and self.jobs_queue.empty():
                    break
            time.sleep(0.1)

    def stop(self):
        with self.lock:
            self.running = False
            for job in self.active_jobs:
                if job.process.is_alive():
                    job.process.terminate()
            self.active_jobs = []
            # Empty the jobs queue
            while not self.jobs_queue.empty():
                self.jobs_queue.get()
        self.monitor_thread.join()

    def available_workers(self):
        with self.lock:
            return self.num_workers - len(self.active_jobs)

    def busy_workers(self):
        with self.lock:
            return len(self.active_jobs)
