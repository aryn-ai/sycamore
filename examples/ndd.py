import sys
import pyarrow.fs
import os

# ruff: noqa: E402
root_dir = os.path.normpath(os.path.dirname(__file__) + "/..")
sys.path.append(root_dir + "/lib/sycamore")

import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.sketcher import SketchDebug


paths = ["s3://aryn-public/ntsb/"]
fsys = pyarrow.fs.S3FileSystem(region="us-east-1", anonymous=True)

tokenizer = HuggingFaceTokenizer("thenlper/gte-small")

ctx = sycamore.init()

ds = (
    ctx.read.binary(paths, binary_format="pdf", filesystem=fsys)
    .partition(partitioner=UnstructuredPdfPartitioner())
    .regex_replace(COALESCE_WHITESPACE)
    .mark_bbox_preset(tokenizer=tokenizer)
    .merge(merger=MarkedMerger())
    .spread_properties(["path", "title"])
    .split_elements(tokenizer=tokenizer, max_tokens=512)
    .explode()
    .sketch()
    .transform(SketchDebug)
)

res = ds.take_all()
print(len(res))
