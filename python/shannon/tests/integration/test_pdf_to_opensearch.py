import shannon
from shannon.tests.config import TEST_DIR


def test_pdf_to_opensearch():
    paths = str(TEST_DIR / "resources/data/pdfs/")
    context = shannon.init()
    ds = context.read.binary(paths, binary_format="pdf") \
        .partition_pdf("bytes", max_partition=256) \
        .sentence_transformer_embed(
        col_name="bytes",
        batch_size=100,
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    ds.show()
