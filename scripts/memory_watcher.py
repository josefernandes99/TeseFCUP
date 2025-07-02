import os
import psutil
import threading
import time
import gc

try:
    import torch
except Exception:
    torch = None


class MemoryWatcher:
    """Background thread that monitors and frees memory."""
    def __init__(self, threshold_percent=80, check_interval=5):
        self.threshold_percent = threshold_percent
        self.check_interval = check_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _free_memory(self):
        before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(
            f"[MemoryWatcher] Freed memory: {before:.2f} MB -> {after:.2f} MB",
            flush=True,
        )

    def _watch(self):
        while not self._stop_event.is_set():
            mem = psutil.virtual_memory()
            if mem.percent >= self.threshold_percent:
                self._free_memory()
            time.sleep(self.check_interval)


def free_unused_memory():
    before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(
        f"[MemoryWatcher] Freed memory: {before:.2f} MB -> {after:.2f} MB",
        flush=True,
    )


def start_memory_watcher(threshold_percent=80, check_interval=5):
    mw = MemoryWatcher(threshold_percent, check_interval)
    mw.start()
    return mw