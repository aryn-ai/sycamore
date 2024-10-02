import sys
import boto3
import pyarrow.fs
import os

# Usage: poetry run python s3_ingest.py s3://<something> [s3://another-thing ...]

# ruff: noqa: E402
sys.path.append("../lib/sycamore")

import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.embed import SentenceTransformerEmbedder

from simple_config import idx_settings, osrch_args, title_template

index = "demoindex0"

if "AWS_SECRET_ACCESS_KEY" in os.environ:
    fsys = None
else:
    print("Attempting to get S3 Credentials")
    sess = boto3.session.Session()
    cred = sess.get_credentials()
    assert cred is not None
    fsys = pyarrow.fs.S3FileSystem(
        access_key=cred.access_key,
        secret_key=cred.secret_key,
        region=sess.region_name,
        session_token=cred.token,
    )

davinci_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
tokenizer = HuggingFaceTokenizer("thenlper/gte-small")

ctx = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)

ds = (
    ctx.read.binary(sys.argv[1:], binary_format="pdf", filesystem=fsys)
    .partition(partitioner=UnstructuredPdfPartitioner())
    .regex_replace(COALESCE_WHITESPACE)
    .extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=davinci_llm, prompt_template=title_template))
    .mark_bbox_preset(tokenizer=tokenizer)
    .merge(merger=MarkedMerger())
    .spread_properties(["path", "title"])
    .split_elements(tokenizer=tokenizer, max_tokens=512)
    .explode()
    .sketch()
    .embed(embedder=SentenceTransformerEmbedder(model_name="thenlper/gte-small", batch_size=100))
)

ds.write.opensearch(
    os_client_args=osrch_args,
    index_name=index,
    index_settings=idx_settings,
)
