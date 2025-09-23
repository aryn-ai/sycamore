from sycamore.connectors.aryn.ArynReader import DocFilter


def test_doc_filter():
    doc_filter = DocFilter(doc_ids=["doc1", "doc2"])
    doc_list = ["doc1", "doc2", "doc3", "doc4"]

    selected_docs = doc_filter.select(doc_list)

    assert selected_docs == ["doc1", "doc2"] or len(selected_docs) == 2
    assert all(doc in doc_list for doc in selected_docs)

    doc_filter = DocFilter(sample_ratio=0.5, seed=42)
    doc_list = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]

    selected_docs = doc_filter.select(doc_list)

    assert len(selected_docs) == 3
    assert all(doc in doc_list for doc in selected_docs)

    doc_filter2 = DocFilter(sample_ratio=0.5, seed=42)
    selected_docs2 = doc_filter2.select(doc_list)
    assert selected_docs == selected_docs2, "Sampling with the same seed should yield the same results"

    doc_filter = DocFilter(sample_ratio=0.01)
    doc_list = ["doc1", "doc2"]

    selected_docs = doc_filter.select(doc_list)

    assert len(selected_docs) == 1


def test_doc_filter_invalid():
    try:
        DocFilter(doc_ids=["doc1"], sample_ratio=0.5)
    except ValueError as e:
        assert str(e) == "Cannot specify both doc_ids and sample_ratio"

    try:
        DocFilter(sample_ratio=1.5)
    except ValueError as e:
        assert str(e) == "sample_ratio must be between 0 and 1"

    try:
        DocFilter(sample_ratio=-0.1)
    except ValueError as e:
        assert str(e) == "sample_ratio must be between 0 and 1"
