from itertools import islice

from sycamore.utils.generate_ray_func import (
    generate_map_function,
)

__all__ = [
    "batched",
    "generate_map_function",
]


def batched(iterable, chunk_size):
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, chunk_size)), list())
