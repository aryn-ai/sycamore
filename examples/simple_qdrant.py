import sys
import os

# ruff: noqa: E402
sys.path.append("../sycamore")

import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import SycamorePartitioner
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.embed import SentenceTransformerEmbedder

from simple_config import title_template

paths = sys.argv[1:]
if not paths:
    raise RuntimeError("No paths supplied.")

model_name = "sentence-transformers/all-MiniLM-L6-v2"

davinci_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value, api_key=os.environ["OPENAI_API_KEY"])
tokenizer = HuggingFaceTokenizer(model_name)

ctx = sycamore.init()

ds = (
    ctx.read.binary(paths, binary_format="pdf")
    .partition(partitioner=SycamorePartitioner(extract_table_structure=True, extract_images=True))
    .regex_replace(COALESCE_WHITESPACE)
    .extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=davinci_llm, prompt_template=title_template))
    .mark_bbox_preset(tokenizer=tokenizer)
    .merge(merger=MarkedMerger())
    .spread_properties(["path", "title"])
    .split_elements(tokenizer=tokenizer, max_tokens=512)
    .explode()
    .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
    .term_frequency(tokenizer=tokenizer, with_token_ids=True)
    .sketch(window=17)
)

ds.write.qdrant(
    {
        "location": "http://localhost:6333",
    },
    {
        "collection_name": "sycamore_example",
        "vectors_config": {
            "size": 384,
            "distance": "Cosine",
        },
    },
)
