from contextlib import contextmanager
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn


@contextmanager
def new_progress(refresh_per_second: float = 1.0):
    """Progress with regular ETA updates and standard columns.

    The progress auto-refreshes once per second to keep ETA visible and
    updated even during longer operations.
    """
    with Progress(
        "[bold cyan]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=refresh_per_second,
        transient=False,
    ) as prog:
        yield prog

