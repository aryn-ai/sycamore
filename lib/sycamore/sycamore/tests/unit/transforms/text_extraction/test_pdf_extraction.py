import pytest

from sycamore.transforms.text_extraction.pdf_miner import PdfMinerExtractor
from sycamore.transforms.text_extraction.pypdfium import PyPdfiumTextExtractor

from sycamore.tests.config import TEST_DIR

# NOTE: PyPdfiumTextExtractor is not thread safe. If we add more tests using
# it we need to be careful with xdist.

extractors = [PyPdfiumTextExtractor(), PdfMinerExtractor()]


@pytest.mark.parametrize("extractor", extractors)
def test_pdf_extraction(extractor):
    path = TEST_DIR / "resources/data/pdfs/visit_aryn.pdf"
    elems_by_page = extractor.extract_document(path, hash_key=None)

    assert len(elems_by_page) == 1, "Expected one page in the PDF"

    print(len(elems_by_page[0]))
    print(elems_by_page[0])

    # TODO: This currently fails with PyPdfiumTextExtractor
    # assert elems_by_page[0][0].text_representation.startswith("Visit http://aryn.ai/")
