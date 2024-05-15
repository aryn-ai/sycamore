import sys

from weaviate.classes.config import Property, ReferenceProperty
from weaviate.client import AdditionalConfig, ConnectionParams
from weaviate.collections.classes.config import Configure, DataType
from weaviate.config import Timeout

# ruff: noqa: E402
sys.path.append("../sycamore")

import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.embed import SentenceTransformerEmbedder

from simple_config import title_template

paths = sys.argv[1:]
if not paths:
    raise RuntimeError("No paths supplied.")

collection = "DemoCollection"
wv_client_args = {
    "connection_params": ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    ),
    "additional_config": AdditionalConfig(timeout=Timeout(init=2, query=45, insert=300)),
}

collection_config_params = {
    "name": collection,
    "description": "A collection to demo data-prep with sycamore",
    "properties": [
        Property(
            name="properties",
            data_type=DataType.OBJECT,
            nested_properties=[
                Property(
                    name="links",
                    data_type=DataType.OBJECT_ARRAY,
                    nested_properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="url", data_type=DataType.TEXT),
                        Property(name="start_index", data_type=DataType.NUMBER),
                    ],
                ),
            ],
        ),
        Property(name="bbox", data_type=DataType.NUMBER_ARRAY),
        Property(name="shingles", data_type=DataType.INT_ARRAY),
    ],
    "vectorizer_config": [Configure.NamedVectors.text2vec_transformers(name="embedding")],
    "references": [ReferenceProperty(name="parent", target_collection=collection)],
}
model_name = "sentence-transformers/all-MiniLM-L6-v2"

davinci_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
tokenizer = HuggingFaceTokenizer(model_name)

ctx = sycamore.init()

ds = (
    ctx.read.binary(paths, binary_format="pdf")
    .partition(partitioner=UnstructuredPdfPartitioner())
    .regex_replace(COALESCE_WHITESPACE)
    .extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=davinci_llm, prompt_template=title_template))
    .mark_bbox_preset(tokenizer=tokenizer)
    .merge(merger=MarkedMerger())
    .spread_properties(["path", "title"])
    .split_elements(tokenizer=tokenizer, max_tokens=512)
)

ds = ds.explode().embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100)).sketch(window=17)

ds.write.weaviate(wv_client_args=wv_client_args, collection_name=collection, collection_config=collection_config_params)
