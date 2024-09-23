import sys

# ruff: noqa: E402
sys.path.append("../sycamore")

import sycamore
from sycamore.functions import TextOverlapChunker
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.transforms.partition import HtmlPartitioner
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.embed import SentenceTransformerEmbedder

from simple_config import idx_settings, osrch_args, title_template

paths = sys.argv[1:]
if not paths:
    raise RuntimeError("No paths supplied.")

index = "demoindex0"

davinci_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
tokenizer = HuggingFaceTokenizer("thenlper/gte-small")

ctx = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)

ds = (
    ctx.read.binary(paths, binary_format="html")
    .partition(
        partitioner=HtmlPartitioner(
            extract_tables=True,
            text_chunker=TextOverlapChunker(chunk_token_count=1000, chunk_overlap_token_count=120),
        )
    )
    .regex_replace(COALESCE_WHITESPACE)
    .extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=davinci_llm, prompt_template=title_template))
    .merge(merger=GreedyTextElementMerger(tokenizer=tokenizer, max_tokens=512))
    .spread_properties(["path", "title"])
    .explode()
    .sketch()
    .embed(embedder=SentenceTransformerEmbedder(model_name="thenlper/gte-small", batch_size=100))
)

ds.write.opensearch(
    os_client_args=osrch_args,
    index_name=index,
    index_settings=idx_settings,
)
