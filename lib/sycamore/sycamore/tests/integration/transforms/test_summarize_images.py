import sycamore
from sycamore.transforms.partition import ArynPartitioner
from sycamore.transforms.summarize_images import SummarizeImages
from sycamore.tests.config import TEST_DIR


def test_summarize_images():
    path = TEST_DIR / "resources/data/pdfs/Ray_page11.pdf"

    context = sycamore.init()
    image_docs = (
        context.read.binary(paths=[str(path)], binary_format="pdf")
        .partition(ArynPartitioner(extract_images=True, use_partitioning_service=False, use_cache=False))
        .transform(SummarizeImages)
        .explode()
        .filter(lambda d: d.type == "Image")
        .take_all()
    )

    assert len(image_docs) == 1
    assert image_docs[0].properties["summary"]["is_graph"]
