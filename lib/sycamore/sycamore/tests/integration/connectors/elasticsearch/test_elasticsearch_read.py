import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.tests.config import TEST_DIR
from sycamore.connectors.common import compare_docs
from elasticsearch import Elasticsearch


def test_to_elasticsearch():
    url = "http://localhost:9201"
    index_name = "test_index-read"
    wait_for_completion = "wait_for"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")

    OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
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
    ctx.read.document(docs).write.elasticsearch(url=url, index_name=index_name, wait_for_completion=wait_for_completion)
    target_doc_id = docs[-1].doc_id if docs[-1].doc_id else ""
    out_docs = ctx.read.elasticsearch(url=url, index_name=index_name, size=len(docs)).take_all()
    query_params = {"term": {"_id": target_doc_id}}
    query_docs = ctx.read.elasticsearch(url=url, index_name=index_name, query=query_params).take_all()
    with Elasticsearch(url) as es_client:
        es_client.indices.delete(index=index_name)
    assert len(out_docs) == len(docs)
    assert len(query_docs) == 1  # exactly one doc should be returned
    assert all(
        compare_docs(original, plumbed)
        for original, plumbed in zip(
            sorted(docs, key=lambda d: d.doc_id or ""), sorted(out_docs, key=lambda d: d.doc_id or "")
        )
    )
