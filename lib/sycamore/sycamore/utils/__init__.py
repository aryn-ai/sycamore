from itertools import islice

__all__ = [
    "batched",
]


def batched(iterable, chunk_size):
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, chunk_size)), list())
