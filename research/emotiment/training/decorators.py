import functools
from typing import Callable


def log_step(fn: Callable):
    """Minimal decorator to show high-level progression.
    Prints only start and end markers so logs answer: What is the program doing now?"""
    name = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[START] {name}")
        result = fn(*args, **kwargs)
        print(f"[END]   {name}")
        return result

    return wrapper

__all__ = ["log_step"]
