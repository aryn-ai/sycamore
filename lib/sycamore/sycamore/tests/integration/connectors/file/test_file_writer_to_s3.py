import sycamore
from sycamore.data.document import Document
from sycamore.functions.document import split_and_convert_to_image
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import UnstructuredPdfPartitioner

import boto3
from PIL import Image as PImage
from pyarrow.fs import S3FileSystem

from io import BytesIO
import os
import os.path
from pathlib import Path
import tempfile
from urllib.parse import urlparse
import uuid

import logging


def get_s3_fs(session):
    credentials = session.get_credentials()

    fs = S3FileSystem(
        secret_key=credentials.secret_key,
        access_key=credentials.access_key,
        region=session.region_name,
        session_token=credentials.token,
    )
    return fs


def render_as_png(doc: Document) -> Document:
    size = tuple(doc.properties["size"])
    mode = doc.properties["mode"]
    assert doc.binary_representation is not None, "Document must have binary representation to render as PNG"
    image = PImage.frombytes(mode=mode, size=size, data=doc.binary_representation)

    png_image = BytesIO()
    image.save(png_image, format="PNG")
    return Document(doc, binary_representation=png_image.getvalue())


def image_page_filename(doc: Document):
    path = Path(doc.properties["path"])
    base_name = ".".join(path.name.split(".")[0:-1])
    page_num = doc.properties["page_number"]
    return f"{base_name}_page_{page_num}.png"


def one_test_convert_to_images(request, exec_mode, tempdir):
    logging.info(f"RUNNING test in exec mode {exec_mode}")
    from sycamore.tests.integration.connectors.file.test_file_writer_to_s3 import render_as_png
    from sycamore.tests.integration.connectors.file.test_file_writer_to_s3 import image_page_filename

    context = sycamore.init(exec_mode=exec_mode)

    paths = str(TEST_DIR / "resources/data/pdfs/")

    image_docset = (
        context.read.binary(paths, binary_format="pdf")
        .partition(partitioner=UnstructuredPdfPartitioner())
        .flat_map(split_and_convert_to_image)
        .map(render_as_png)
        .materialize(path=tempdir, source_mode=sycamore.MATERIALIZE_USE_STORED)
    )

    logging.info("Running conversion")
    num_pages = image_docset.count()

    session = boto3.session.Session()

    s3_url = os.environ["SYCAMORE_S3_TEMP_PATH"]
    test_path = str(uuid.uuid4())
    out_path = os.path.join(s3_url, request.node.originalname, test_path)

    logging.info("Writing files to S3")
    image_docset.write.files(path=out_path, filesystem=get_s3_fs(session), filename_fn=image_page_filename)

    logging.info("Reading files from S3")
    read_docs = context.read.binary([out_path], filesystem=get_s3_fs(session), binary_format="png").take_all()
    assert len(read_docs) == num_pages

    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    parsed_s3_url = urlparse(out_path)
    bucket = parsed_s3_url.netloc
    s3_path = parsed_s3_url.path

    if s3_path.startswith("/"):
        s3_path = s3_path.lstrip("/")
    if not s3_path.endswith("/"):
        s3_path = s3_path + "/"

    print("bucket", bucket)
    print("s3_path", s3_path)

    keys = []
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=s3_path)
    for page in page_iterator:
        for k in page["Contents"]:
            # Because "prefixes" are objects in S3, they show up in listing.
            if not k["Key"].endswith("/"):
                keys.append(k["Key"])

    assert len(keys) == num_pages

    logging.info("Deleting files from S3")
    # Cleanup if the test succeeded. If the test fails, we wan to leave the objects for a while
    # for debugging. It's not a crisis if these calls fail because we have a lifecycle policy on
    # bucket to clean up old objects.
    s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys]})

    s3.delete_object(Bucket=bucket, Key=s3_path)


def test_convert_to_images(request):
    with tempfile.TemporaryDirectory() as tempdir:
        one_test_convert_to_images(request, sycamore.EXEC_LOCAL, tempdir)
        one_test_convert_to_images(request, sycamore.EXEC_RAY, tempdir)
