import os

from itertools import islice

__all__ = [
    "batched",
    "use_cuda",
]


def batched(iterable, chunk_size):
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, chunk_size)), list())


def use_cuda() -> bool:
    if os.environ.get("DISABLE_CUDA", 0) == "1":
        return False

    import torch.cuda

    return torch.cuda.is_available()
