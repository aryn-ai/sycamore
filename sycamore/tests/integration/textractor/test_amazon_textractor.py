import boto3

import sycamore
from sycamore.execution.transforms.partition import UnstructuredPdfPartitioner
from sycamore.execution.transforms.table_extraction import TextractTableExtractor


def get_s3_fs():
    session = boto3.session.Session()
    credentials = session.get_credentials()
    from pyarrow.fs import S3FileSystem

    fs = S3FileSystem(
        secret_key=credentials.secret_key,
        access_key=credentials.access_key,
        region=session.region_name,
        session_token=credentials.token,
    )
    return fs


class TestTextExtraction:
    def test_extract_tables(self):
        context = sycamore.init()
        docset = context.read.binary("s3://aryn-textract/10q-excerpt.pdf", binary_format="pdf", filesystem=get_s3_fs())
        document = docset.take(1)[0]
        table_extractor = TextractTableExtractor(region_name="us-east-1")
        tables = table_extractor.extract_tables(document)
        assert len(tables) == 2

    def test_docset_extract_tables(self):
        context = sycamore.init()
        docset = context.read.binary("s3://aryn-textract/10q-excerpt.pdf", binary_format="pdf", filesystem=get_s3_fs())
        docset = docset.partition(partitioner=UnstructuredPdfPartitioner())
        priori = docset.take(1)[0]
        docset = docset.extract_tables(region_name="us-east-1")
        post = docset.take(1)[0]
        assert len(post.elements) == len(priori.elements) + 2
