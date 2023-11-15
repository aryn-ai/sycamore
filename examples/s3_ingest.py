import sys
import boto3
import pyarrow.fs

# ruff: noqa: E402
sys.path.append("../sycamore")

import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import GreedyTextElementMerger, MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.embed import SentenceTransformerEmbedder

from simple_config import idx_settings, osrch_args, title_template

paths = sys.argv[1:]
if not paths:
    raise RuntimeError("No S3 URLs supplied.")

index = "demoindex0"

sess = boto3.session.Session()
cred = sess.get_credentials()
fsys = pyarrow.fs.S3FileSystem(
    access_key=cred.access_key,
    secret_key=cred.secret_key,
    region=sess.region_name,
    session_token=cred.token,
)

davinci_llm = OpenAI(OpenAIModels.TEXT_DAVINCI.value)
tokenizer = HuggingFaceTokenizer("thenlper/gte-small")

ctx = sycamore.init()

ds = (
    ctx.read.binary(paths, binary_format="pdf", filesystem=fsys)
    .partition(partitioner=UnstructuredPdfPartitioner())
    .regex_replace(COALESCE_WHITESPACE)
    .extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=davinci_llm, prompt_template=title_template))
    .mark_bbox_preset(tokenizer=tokenizer)
    .merge(merger=MarkedMerger())
    .spread_properties(["path", "title"])
    .explode()
    .embed(embedder=SentenceTransformerEmbedder(model_name="thenlper/gte-small", batch_size=100))
)

ds.write.opensearch(
    os_client_args=osrch_args,
    index_name=index,
    index_settings=idx_settings,
)
