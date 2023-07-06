import sycamore
from sycamore.tests.config import TEST_DIR


def test_pdf_to_opensearch():
    paths = str(TEST_DIR / "resources/data/pdfs/")
    context = sycamore.init()
    ds = context.read.binary(paths, binary_format="pdf") \
        .unstructured_partition(max_partition=256) \
        .explode() \
        .sentence_transformer_embed(
        batch_size=100,
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    ds.show()
