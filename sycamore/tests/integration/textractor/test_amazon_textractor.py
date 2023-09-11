import boto3

import sycamore
from execution.transforms import PdfPartitionerOptions
from execution.transforms.table_extraction import TextractorTableExtractor


def get_s3_fs():
    session = boto3.session.Session(profile_name="dev-admin")
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
        table_extractor = TextractorTableExtractor(profile_name="dev-admin")
        tables = table_extractor._extract_tables(document)
        assert len(tables) == 2

    def test_docset_extract_tables(self):
        context = sycamore.init()
        docset = context.read.binary("s3://aryn-textract/10q-excerpt.pdf", binary_format="pdf", filesystem=get_s3_fs())
        docset = docset.partition(options=PdfPartitionerOptions())
        priori = docset.take(1)[0]
        docset = docset.extract_tables(profile_name="dev-admin")
        post = docset.take(1)[0]
        assert len(post.elements) == len(priori.elements) + 2
