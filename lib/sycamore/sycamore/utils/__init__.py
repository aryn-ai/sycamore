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


def choose_device(want: Optional[str], *, detr=False) -> str:
    if os.environ.get("DISABLE_GPU") == "1":
        return "cpu"
    if want:
        return want

    import torch.cuda

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"  # !!! as of 6/17/2024 on macs cpu is faster than mps

    import torch.backends.mps

    if torch.backends.mps.is_available():
        if detr:
            import torch

            if torch.__version__ < "2.3":
                return "cpu"  # Older torch doesn't support DETR on MPS
        return "mps"

    return "cpu"
