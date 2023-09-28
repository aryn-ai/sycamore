import boto3

import sycamore
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.extract_table import TextractTableExtractor


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
        tables = table_extractor._extract(document)
        assert len(tables) == 2

    def test_docset_extract_tables(self):
        context = sycamore.init()
        docset = context.read.binary("s3://aryn-textract/10q-excerpt.pdf", binary_format="pdf", filesystem=get_s3_fs())
        docset_no_tables = docset.partition(partitioner=UnstructuredPdfPartitioner())
        docset_with_tables = docset.partition(
            partitioner=UnstructuredPdfPartitioner(), table_extractor=TextractTableExtractor(region_name="us-east-1")
        )

        total_elements = len(docset_no_tables.take(1)[0].elements)
        total_elements_elements_with_tables = len(docset_with_tables.take(1)[0].elements)
        assert total_elements_elements_with_tables == total_elements + 2
