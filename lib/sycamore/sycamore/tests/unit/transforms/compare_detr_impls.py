from sycamore.utils.deep_eq import deep_eq


def compare_batched_sequenced(partitioner, path, **kwargs):
    batched = partitioner._partition_pdf_batched_named(path, **kwargs)
    with open(path, "rb") as f:
        sequenced = partitioner._partition_pdf_sequenced(f, **kwargs)

    assert deep_eq(batched, sequenced)
    return batched


if __name__ == "__main__":
    import sys
    from sycamore.transforms.detr_partitioner import ArynPDFPartitioner

    assert len(sys.argv) == 2, "Usage: cmd <path>"
    s = ArynPDFPartitioner("Aryn/deformable-detr-DocLayNet")
    print(f"Comparing processing of {sys.argv[1]}")
    p = compare_batched_sequenced(s, sys.argv[1])
    print(f"Compared {len(p)} pages")
