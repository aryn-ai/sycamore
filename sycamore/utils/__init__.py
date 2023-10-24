from itertools import islice

from sycamore.utils.generate_ray_func import (
    generate_map_function,
    generate_map_class,
    generate_map_class_from_callable,
    generate_flat_map_function,
    generate_flat_map_class,
    generate_map_batch_function,
    generate_map_batch_filter_function,
    generate_map_batch_class,
    generate_map_batch_class_from_callable,
)

__all__ = [
    "batched",
    "generate_map_function",
    "generate_map_class",
    "generate_map_class_from_callable",
    "generate_flat_map_function",
    "generate_flat_map_class",
    "generate_map_batch_function",
    "generate_map_batch_filter_function",
    "generate_map_batch_class",
    "generate_map_batch_class_from_callable",
]


def batched(iterable, chunk_size):
    iterator = iter(iterable)
    return iter(lambda: list(islice(iterator, chunk_size)), list())
