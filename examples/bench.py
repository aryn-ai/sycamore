# This is the current base speed performance benchmarking script.
# For actual timing runs, it may need to be modified to set the
# numbers of CPUs, GPUs, batches, devices, etc.  Other Sycamore
# code may also need to be changed to adjust actor pools.  The
# script should be run similar to this:
#
# TIMETRACE=/tmp/tt poetry run python examples/bench.py

import sys
import pyarrow.fs

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

from simple_config import idx_settings, osrch_args, title_template

paths = ["s3://aryn-public/ntsb/"]
index = "demoindex0"
fsys = pyarrow.fs.S3FileSystem(region="us-east-1", anonymous=True)

davinci_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
tokenizer = HuggingFaceTokenizer("thenlper/gte-small")

ctx = sycamore.init()

ds = (
    ctx.read.binary(paths, binary_format="pdf", filesystem=fsys)
    .partition(partitioner=SycamorePartitioner(extract_table_structure=True, extract_images=True))
    .regex_replace(COALESCE_WHITESPACE)
    .extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=davinci_llm, prompt_template=title_template))
    .mark_bbox_preset(tokenizer=tokenizer)
    .merge(merger=MarkedMerger())
    .spread_properties(["path", "title"])
    .split_elements(tokenizer=tokenizer, max_tokens=512)
    .explode()
    .sketch()
    .embed(embedder=SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=100))
)

ds.write.opensearch(
    os_client_args=osrch_args,
    index_name=index,
    index_settings=idx_settings,
)
