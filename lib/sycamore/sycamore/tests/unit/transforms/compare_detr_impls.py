from sycamore.utils.cache import Cache
from sycamore.utils.deep_eq import deep_eq


def check_partition(partitioner, path, **kwargs):
    with open(path, "rb") as f:
        hash_key = Cache.get_hash_context(f.read()).hexdigest()
    batched = partitioner._partition_pdf_batched_named(path, hash_key, **kwargs)
    assert batched is not None
    return batched


def check_table_extraction(partitioner, path, **kwargs):
    with open(path, "rb") as f:
        hash_key = Cache.get_hash_context(f.read()).hexdigest()
    batched = partitioner._partition_pdf_batched_named(path, hash_key, **kwargs)
    assert all(
        (
            d.tokens is not None and all("bbox" in token and isinstance(token["bbox"], list) for token in d.tokens)
            if d.type == "table"
            else True
        )
        for batched_list in batched
        for d in batched_list
    )
    return batched


if __name__ == "__main__":
    import sys
    from sycamore.transforms.detr_partitioner import ArynPDFPartitioner

    assert len(sys.argv) == 2, "Usage: cmd <path>"
    s = ArynPDFPartitioner("Aryn/deformable-detr-DocLayNet")
    print(f"Comparing processing of {sys.argv[1]}")
    p = check_partition(s, sys.argv[1])
    print(f"Compared {len(p)} pages")
