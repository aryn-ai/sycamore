import sycamore.utils.jupyter as suj
from sycamore.data import Document


def test_init_viewpdf():
    def foo(doc: Document) -> str:
        return "foo"

    def bar(doc: Document) -> str:
        return "bar"

    suj.init_viewpdf(foo, bar)
    assert suj._doc_to_url is foo  # type: ignore
    assert suj._doc_to_display_name is bar  # type: ignore

    delattr(suj, "_doc_to_url")
    delattr(suj, "_doc_to_display_name")
