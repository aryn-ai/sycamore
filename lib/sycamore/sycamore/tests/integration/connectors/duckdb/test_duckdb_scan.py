import os

import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.tests.config import TEST_DIR
from sycamore.connectors.common import compare_docs


def test_duckdb_scan():
    table_name = "duckdb_table"
    db_url = "tmp.db"
    paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = HuggingFaceTokenizer(model_name)

    ctx = sycamore.init()

    docs = (
        ctx.read.binary(paths, binary_format="pdf")
        .partition(partitioner=UnstructuredPdfPartitioner())
        .regex_replace(COALESCE_WHITESPACE)
        .mark_bbox_preset(tokenizer=tokenizer)
        .merge(merger=MarkedMerger())
        .spread_properties(["path"])
        .split_elements(tokenizer=tokenizer, max_tokens=512)
        .explode()
        .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
        .sketch(window=17)
        .take_all()
    )
    ctx.read.document(docs).write.duckdb(db_url=db_url, table_name=table_name, dimensions=384)

    out_docs = ctx.read.duckdb(db_url=db_url, table_name=table_name).take_all()
    try:
        os.unlink(db_url)
    except Exception as e:
        print(f"Error deleting {db_url}: {e}")
    assert len(out_docs) == len(docs)
    assert all(
        compare_docs(original, plumbed)
        for original, plumbed in zip(
            sorted(docs, key=lambda d: d.doc_id or ""), sorted(out_docs, key=lambda d: d.doc_id or "")
        )
    )
