import os
from typing import Optional

from itertools import islice

__all__ = [
    "batched",
    "choose_device",
]


def batched(iterable, chunk_size):
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, chunk_size)), list())


def choose_device(want: Optional[str]) -> str:
    if os.environ.get("DISABLE_GPU") == "1":
        return "cpu"
    if want:
        return want

    import torch.cuda

    if torch.cuda.is_available():
        return "cuda"

    import torch.backends.mps

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"
