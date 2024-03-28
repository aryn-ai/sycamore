from sycamore.transforms.partition import SYCAMORE_DETR_MODEL, SycamorePartitioner
import sycamore
from sycamore.tests.config import TEST_DIR


def test_detr_ocr():
    path = TEST_DIR / "resources/data/pdfs/Transformer.pdf"

    context = sycamore.init()

    # TODO: The title on the paper is recognized as a section header rather than a page header at the moment.
    # The test will need to be updated if and when that changes.
    docs = (
        context.read.binary(paths=[str(path)], binary_format="pdf")
        .partition(SycamorePartitioner(SYCAMORE_DETR_MODEL, use_ocr=True))
        .explode()
        .filter(lambda doc: "page_number" in doc.properties and doc.properties["page_number"] == 1)
        .filter(lambda doc: doc.type == "Section-header")
        .take_all()
    )

    assert "Attention Is All You Need" in set(str(d.text_representation).strip() for d in docs)
