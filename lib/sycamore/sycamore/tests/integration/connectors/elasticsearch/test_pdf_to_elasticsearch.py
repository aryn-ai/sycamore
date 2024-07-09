import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.tests.config import TEST_DIR
from elasticsearch import Elasticsearch


def test_to_elasticsearch():
    url = "http://localhost:9200"
    client = Elasticsearch("http://localhost:9200")

    # pass client object to info() method
    elastic_info = Elasticsearch.info(client)
    print(elastic_info)
    index_name = "test_index"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    paths = str(TEST_DIR / "resources/data/pdfs/")

    OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
    tokenizer = HuggingFaceTokenizer(model_name)

    ctx = sycamore.init()

    ds = (
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
    )
    ds.write.elasticsearch(url=url, index_name=index_name)
