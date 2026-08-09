# fmt: off
import sys
from typing import BinaryIO
from itertools import repeat
from collections.abc import Iterator

import boto3
import pyarrow.fs

import aryn_sdk.partition
import sycamore
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.sketcher import Sketcher
from sycamore.connectors.duckdb.duckdb_writer import (
    DuckDBWriterClientParams,
    DuckDBWriterTargetParams,
    DuckDBWriter,
)

###############################################################################

def iterInputs(inputs: list[str], aws_sess = None) -> Iterator[BinaryIO]:
    fs_s3 = None
    for input in inputs:
        if input.startswith("s3://"):
            path = input[5:]
            if not fs_s3:
                cred = aws_sess.get_credentials()
                fs_s3 = pyarrow.fs.S3FileSystem(
                    access_key=cred.access_key,
                    secret_key=cred.secret_key,
                    region=aws_sess.region_name,
                    session_token=cred.token,
                )
            fsys = fs_s3
        else:
            if input.startswith("file://"):
                path = input[7:]
            else:
                path = input
            fsys = pyarrow.fs.LocalFileSystem()

        info = fsys.get_file_info(path)
        if info.type == pyarrow.fs.FileType.File:
            infos = [info]
        else:
            infos = fsys.get_file_info(
                pyarrow.fs.FileSelector(path, recursive=True)
            )
        for info in infos:
            strm = fsys.open_input_stream(info.path)
            yield (input, strm)

###############################################################################

inputs = [
    "s3://aryn-public/cccmad-tiny/2019/006965487_National Student Nurses Association.pdf.pdf",
    "s3://aryn-public/cccmad-tiny/2011/",
    "file:///home/alex/load/pdfs/ntsb0.pdf",
    "/home/alex/load/pdfs/ca_san_luis_obispo_21_22_adopted.pdf",
]

aws_sess = boto3.session.Session()
embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")

ddb_writer = DuckDBWriter(
    None,  # FIXME default plan to None
    DuckDBWriterClientParams(),
    DuckDBWriterTargetParams(dimensions=384),  # FIXME get from embedder
)

for fn, strm in iterInputs(inputs, aws_sess):
    res = aryn_sdk.partition.partition_file(strm)
    bigdoc = sycamore.data.document.Document(res)
    docs = sycamore.transforms.Explode.explode(bigdoc)
    for idx in range(len(docs)):
        docs[idx].doc_id = f"{fn}-{idx}"  # FIXME make explode do this
    docs = list(map(Sketcher.sketcher, docs, repeat(17), repeat(16)))  # FIXME default numeric params
    docs = embedder.generate_embeddings(docs)
    ddb_writer.write_docs(docs)
