from contextlib import contextmanager
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)

console = Console(highlight=False, soft_wrap=True)


class _SimpleTask:
    def __init__(self, description: str, total: int | None):
        self.description = description
        self.total = total
        self.completed = 0
        self.start_time = time.time()
        self.started = False
        self.finished = False
        self._log_start()

    def advance(self, amount: int):
        if amount <= 0:
            return
        self.completed += amount
        if self.total:
            percent = min(100, int((self.completed / self.total) * 100))
            if percent >= 100:
                self._log_complete()
        else:
            # For indeterminate tasks we only log start/finish
            pass

    def complete(self):
        if not self.finished:
            self._log_complete()

    def _log_start(self):
        elapsed = time.time() - self.start_time
        if self.total:
            console.print(
                f"{self.description:<32} started (0/{self.total})",
                style="cyan",
                highlight=False,
            )
        else:
            console.print(
                f"{self.description:<32} started",
                style="cyan",
                highlight=False,
            )
        self.started = True

    def _log_complete(self):
        if self.finished:
            return
        elapsed = time.time() - self.start_time
        if self.total:
            console.print(
                f"{self.description:<32} completed ({self.total}/{self.total}) • {elapsed:5.1f}s",
                style="cyan",
                highlight=False,
            )
        else:
            console.print(
                f"{self.description:<32} completed • {elapsed:5.1f}s",
                style="cyan",
                highlight=False,
            )
        self.finished = True


class _SimpleProgress:
    def __init__(self):
        self._tasks: dict[int, _SimpleTask] = {}
        self._next_id = 1

    def add_task(self, description: str, total: int | None = None):
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = _SimpleTask(description, total)
        return task_id

    def update(self, task_id: int, advance: int = 1):
        task = self._tasks.get(task_id)
        if task:
            task.advance(advance)

    def complete(self, task_id: int):
        task = self._tasks.get(task_id)
        if task:
            task.complete()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for task in self._tasks.values():
            task.complete()


@contextmanager
def new_progress(refresh_per_second: float = 1.0):
    """Return a progress helper that adapts to the environment."""
    if console.is_terminal:
        columns = [
            TextColumn("{task.description:<30}", style="cyan"),
            BarColumn(bar_width=24),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        with Progress(*columns, console=console, refresh_per_second=refresh_per_second, transient=False) as prog:
            yield prog
    else:
        yield _SimpleProgress()


def print_section(title: str, subtitle: str | None = None):
    console.print(Panel(title, subtitle=subtitle, expand=False, border_style="bright_blue"))


__all__ = ["new_progress", "console", "print_section"]

