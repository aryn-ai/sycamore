from io import IOBase
from typing import cast

from sycamore.utils.cache_manager import CacheManager
from sycamore.utils.deep_eq import deep_eq


def compare_batched_sequenced(partitioner, path, **kwargs):
    with open(path, "rb") as f:
        sequenced = partitioner._partition_pdf_sequenced(f, **kwargs)
        hash_key = CacheManager.get_hash_key(f.read())
    batched = partitioner._partition_pdf_batched_named(path, hash_key, **kwargs)
    assert deep_eq(batched, sequenced)
    return batched


if __name__ == "__main__":
    import sys
    from sycamore.transforms.detr_partitioner import SycamorePDFPartitioner

    assert len(sys.argv) == 2, "Usage: cmd <path>"
    s = SycamorePDFPartitioner("Aryn/deformable-detr-DocLayNet")
    print(f"Comparing processing of {sys.argv[1]}")
    p = compare_batched_sequenced(s, sys.argv[1])
    print(f"Compared {len(p)} pages")
