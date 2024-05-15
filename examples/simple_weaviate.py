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

"""
{'properties': {'filename': '',
                'filetype': 'application/pdf',
                'parent_id': '31d568869366fcd6b0c2ea984c045351',
                'page_numbers': [11],
                'page_number': 11,
                'links': [],
                'element_id': 'e274015adef0b32a467da587b0767bf8',
                'path': 'lib/sycamore/sycamore/tests/resources/data/pdfs/Transformer.pdf',
                'title': 'Attention Is All You Need'},
 'binary_representation': b'<149 bytes>',
 'text_representation': '[23] Romain Paulus, Caiming Xiong, and Richard '
                        'Socher. A deep reinforced model for abstractive\n'
                        'summarization. arXiv preprint arXiv:1705.04304, '
                        '2017.\n',
 'bbox': (0.17647058823529413,
          0.16884889090909086,
          0.8235248676274509,
          0.19520192121212124),
 'elements': [],
 'doc_id': '5c248bce-122c-11ef-a5d4-3a049e7f7082',
 'parent_id': '4b5c7bd0-122c-11ef-82bf-3a049e7f7082',
 'embedding': '<384 floats>',
 'shingles': [3895770718133863,
              177162244638153298,
              584376959995933244,
              629082248197781499,
              681250792601014459,
              887451002489775145,
              936202876338592785,
              997937039503815988,
              1080265256187218690,
              1100561500964758549,
              1130572111381058751,
              1149317895372858111,
              1216481584357509483,
              1255963861059033531,
              1267688795828793018,
              1302217756253739570]}
 """

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
                Property(name="filetype", data_type=DataType.TEXT),
                Property(name="filename", data_type=DataType.TEXT),
                Property(name="page_number", data_type=DataType.INT),
                Property(name="page_numbers", data_type=DataType.INT_ARRAY),
                Property(
                    name="links",
                    data_type=DataType.OBJECT_ARRAY,
                    nested_properties=[
                        Property(name="test", data_type=DataType.TEXT),
                        Property(name="url", data_type=DataType.TEXT),
                        Property(name="start_index", data_type=DataType.INT),
                    ],
                ),
                Property(name="path", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
            ],
        ),
        Property(name="text_representation", data_type=DataType.TEXT),
        Property(name="type", data_type=DataType.TEXT),
        Property(name="bbox", data_type=DataType.INT_ARRAY),
        Property(name="shingles", data_type=DataType.INT_ARRAY),
    ],
    "vectorizer_config": [Configure.NamedVectors.none(name="embedding")],
    "references": [ReferenceProperty(name="parent_id", target_collection=collection)],
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

ds.show(limit=1000, truncate_length=500)

ds = (
    ds.explode()
    .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
    .sketch(window=17)
    .filter(lambda doc: doc.text_representation is not None)
)

ds.show(limit=1000, truncate_length=500)
ds.write.weaviate(wv_client_args=wv_client_args, collection_name=collection, collection_config=collection_config_params)
