import boto3

import sycamore
from sycamore.data import Element
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.extract_table import TextractTableExtractor, CachedTextractTableExtractor


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
        docset_no_tables = docset.partition(partitioner=UnstructuredPdfPartitioner())
        docset_with_tables = docset.partition(
            partitioner=UnstructuredPdfPartitioner(), table_extractor=TextractTableExtractor(region_name="us-east-1")
        )

        total_elements = len(docset_no_tables.take(1)[0].elements)
        total_elements_elements_with_tables = len(docset_with_tables.take(1)[0].elements)
        assert total_elements_elements_with_tables == total_elements + 2

    def test_cached_textractor(self):
        context = sycamore.init()
        docset = context.read.binary(
            "s3://aryn-benchmark/data/integration/cachedtextractor/pdf/assembled_sram_table_doc.pdf",
            binary_format="pdf",
            filesystem=get_s3_fs(),
        )
        document = docset.take(1)[0]
        element1 = Element({"type": "Table", "properties": {"page_number": 1}})
        element2 = Element({"type": "Table", "properties": {"page_number": 2}})
        element3 = Element({"type": "Table", "properties": {"page_number": 6}})
        element4 = Element({"type": "Table", "properties": {"page_number": 7}})
        document.elements = [element1, element2, element3, element4]

        cached_extractor = CachedTextractTableExtractor(
            s3_cache_location="s3://aryn-benchmark/data/integration/cachedtextractor/cache", region_name="us-east-1"
        )

        document = cached_extractor.extract_tables(document)

        assert len(document.elements) == 13
        assert document.elements[12].properties["page_number"] == 7
