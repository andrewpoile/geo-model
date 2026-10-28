"""Map a task over items in worker processes, the task sent to each worker
once as it starts rather than with every item, since a task binds every
student sample and its distance to every school."""

import contextlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor

# The task a worker process maps, set as the worker starts.
_task: Callable


def _set_task(task: Callable) -> None:
    global _task
    _task = task


def _run_task(item):
    return _task(item)


@contextlib.contextmanager
def task_map(task: Callable, workers: int) -> Iterator[Callable[[Iterable], Iterator]]:
    """`map` of `task`, in order, over `workers` processes when more than one.

    Args:
        task (Callable): Called on each item, picklable.

        workers (int): Processes to map in; 1 or fewer maps in this one.

    Yields:
        Callable[[Iterable], Iterator]: Maps `task` over the items given.
    """
    if workers <= 1:
        yield lambda items: map(task, items)
        return
    with ProcessPoolExecutor(workers, initializer=_set_task, initargs=(task,)) as pool:
        yield lambda items: pool.map(_run_task, items)
