import pytest
from unittest.mock import Mock
from sycamore.utils.similarity import make_element_sorter_fn


def test_make_element_sorter_fn_no_similarity_query():
    sorter_fn = make_element_sorter_fn("test_field", None, Mock())
    assert sorter_fn({}) is None


def test_make_element_sorter_fn_no_similarity_scorer():
    with pytest.raises(AssertionError, match="Similarity sorting requires a scorer"):
        make_element_sorter_fn("test_field", "query", None)
