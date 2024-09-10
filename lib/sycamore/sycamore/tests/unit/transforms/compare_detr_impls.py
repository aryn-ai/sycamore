from sycamore.utils.cache import Cache
from sycamore.utils.deep_eq import deep_eq


def compare_batched_sequenced(partitioner, path, **kwargs):
    with open(path, "rb") as f:
        sequenced = partitioner._partition_pdf_sequenced(f, **kwargs)
        hash_key = Cache.get_hash_context(f.read()).hexdigest()
    batched = partitioner._partition_pdf_batched_named(path, hash_key, **kwargs)
    assert deep_eq(batched, sequenced)
    return batched


def check_table_extraction(partitioner, path, **kwargs):
    with open(path, "rb") as f:
        sequenced = partitioner._partition_pdf_sequenced(f, **kwargs)
        hash_key = Cache.get_hash_context(f.read()).hexdigest()
    batched = partitioner._partition_pdf_batched_named(path, hash_key, **kwargs)
    assert deep_eq(batched, sequenced)
    assert all(
        (
            d.tokens is not None and "bbox" in d.tokens and isinstance(d.tokens["bbox"], list)
            if d.type == "table"
            else True
        )
        for batched_list in batched
        for d in batched_list
    )
    assert all(
        (
            d.tokens is not None and "bbox" in d.tokens and isinstance(d.tokens["bbox"], list)
            if d.type == "table"
            else True
        )
        for sequenced_list in sequenced
        for d in sequenced_list
    )
    return batched


if __name__ == "__main__":
    import sys
    from sycamore.transforms.detr_partitioner import ArynPDFPartitioner

    assert len(sys.argv) == 2, "Usage: cmd <path>"
    s = ArynPDFPartitioner("Aryn/deformable-detr-DocLayNet")
    print(f"Comparing processing of {sys.argv[1]}")
    p = compare_batched_sequenced(s, sys.argv[1])
    print(f"Compared {len(p)} pages")
