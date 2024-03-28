from abc import abstractmethod, ABC
from collections import OrderedDict
import io
from typing import Optional

import boto3
from botocore.exceptions import ClientError
import json
import PyPDF2
from textractor import Textractor
from textractor.data.constants import TextractFeatures

from sycamore.data import BoundingBox, Document, Element


class MissingS3UploadPath(Exception):
    "Raised when an S3 upload path is needed but one wasn't provided"
    pass


class TableExtractor(ABC):
    @abstractmethod
    def extract_tables(self, document: Document) -> Document:
        pass


class TextractTableExtractor(TableExtractor):
    """
    TextractTableExtractor utilizes Amazon Textract to extract tables from documents.

    This class inherits from TableExtractor and is designed for extracting tables from documents using Amazon Textract,
    a cloud-based document text and data extraction service from AWS.

    Args:
        profile_name: The AWS profile name to use for authentication. Default is None.
        region_name: The AWS region name where the Textract service is available.
        kms_key_id: The AWS Key Management Service (KMS) key ID for encryption.

    Example:
         .. code-block:: python

            table_extractor = TextractTableExtractor(profile_name="my-profile", region_name="us-east-1")

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner(), table_extractor=table_extractor)
    """

    def __init__(
        self,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        kms_key_id: str = "",
        s3_upload_root: str = "",
    ):
        self._profile_name = profile_name
        self._region_name = region_name
        self._kms_key_id: str = kms_key_id
        self._s3_upload_root: str = s3_upload_root

    def _extract(self, document: Document) -> list[Element]:
        # https://docs.aws.amazon.com/textract/latest/dg/API_BoundingBox.html
        def bbox_to_coord(bbox):
            return bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height

        extractor = Textractor(self._profile_name, self._region_name, self._kms_key_id)
        path = document.properties["path"]
        if path.startswith("s3://"):  # if document is already in s3, don't upload it again
            result = extractor.start_document_analysis(document.properties["path"], TextractFeatures.TABLES)
        elif not self._s3_upload_root.startswith("s3://"):
            raise MissingS3UploadPath()
        else:
            # TODO: https://github.com/aryn-ai/sycamore/issues/173 - implement content-hash uploading
            # If we manually upload based on a hash, we can avoid repeated uploads and storage
            # of the same file.
            tmp_path = path
            if not tmp_path.startswith("/"):
                tmp_path = "/" + path

            # os.path.join("s3://foo", "/abc") -> "/abc"; which is not what we want.
            dest = self._s3_upload_root + tmp_path

            result = extractor.start_document_analysis(
                document.properties["path"], TextractFeatures.TABLES, s3_upload_path=dest
            )

        # map page_number -> list of tables on that page number
        all_tables = []
        for table in result.tables:
            element = Element()
            element.type = "Table"
            properties = element.properties
            properties["boxes"] = []
            properties["id"] = table.id
            properties["page_number"] = table.page
            element.properties = properties

            if table.title:
                element.text_representation = table.title.text + "\n"
            else:
                element.text_representation = ""

            element.text_representation = element.text_representation + table.to_csv() + "\n"
            element.bbox = BoundingBox(*bbox_to_coord(table.bbox))

            for footer in table.footers:
                element.text_representation = element.text_representation + footer.text + "\n"

            all_tables.append(element)

        return all_tables

    def extract_tables(self, document: Document) -> Document:
        tables = self._extract(document)
        document.elements = document.elements + tables
        return document


class CachedTextractTableExtractor(TableExtractor):
    """
    TextractTableExtractor with S3 based cache support

    For each document, CachedTextractTableExtractor follows a series of steps for table extraction
     1. if cache exists for current document, get from cache and return, otherwise, go to step 2
     2. if run_full_textract is enabled, call textractor on the whole document and go to step 5; otherwise, go to step 3
     3. if any element in the document is not marked table, return, otherwise, go to step 4.
     4. clip pages which contain tables and run table extraction using textractor
     5. update cache accordingly based on textractor result, return updated document

    Cached table is in Json format as below:
    {
      "tables": [
        {
          "page_number": xx,
          "title": "xxxx",
          "content": "xxx"
          "bbox": [xxx, xxx, xxx, xxx],
          "footers": ["xxx", "xxx"]
        },
        ...
      ]
    }
    bbox format is in [left/width, top/height, right/width, bottom/height]
    """

    def __init__(
        self,
        s3_cache_location,
        run_full_textract: bool = False,
        s3_textract_upload_path: str = "",
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        kms_key_id: str = "",
    ):
        self._s3_cache_location = s3_cache_location
        self._run_full_textract = run_full_textract
        self._s3_textract_upload_path = s3_textract_upload_path
        self._profile_name = profile_name
        self._region_name = region_name
        self._kms_key_id = kms_key_id

    @staticmethod
    def _parse(json_data):
        """
        Parse json format cache into a list of table elements
        """
        tables = []
        for table in json_data["tables"]:
            element = Element()
            element.type = "Table"
            properties = element.properties
            properties["page_number"] = table["page_number"]
            element.properties = properties

            text_list = []
            if table.get("title"):
                text_list.append(table["title"])
            text_list.append(table["content"])
            element.bbox = BoundingBox(*table["bbox"])
            text_list.extend(table.get("footers", []))

            element.text_representation = "\n".join(text_list)
            tables.append(element)
        return tables

    @staticmethod
    def _wrap(result, table_pages):
        """
        Wrap textract return data into json format
        """
        cached_tables = []
        for table in result.tables:
            t = {
                "content": table.to_csv(),
                "page_number": table_pages[table.page - 1] if table_pages else table.page,
                "bbox": [table.bbox.x, table.bbox.y, table.bbox.x + table.bbox.width, table.bbox.y + table.bbox.height],
            }
            if table.title:
                t["title"] = table.title.text
            footers = [footer.text for footer in table.footers]
            if footers:
                t["footers"] = footers
            cached_tables.append(t)
        return {"tables": cached_tables}

    def _get_table(self, s3, cache_id) -> Optional[list[Element]]:
        """Get cache from S3"""
        try:
            parts = self._s3_cache_location.replace("s3://", "").strip("/").split("/", 1)
            bucket = parts[0]
            key = "/".join([parts[1], cache_id]) if len(parts) == 2 else cache_id
            response = s3.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            else:
                raise

        tables = self._parse(json.loads(response["Body"].read()))
        return tables

    def _put_table(self, s3, cache_id: str, table: dict):
        """Put table into S3"""
        parts = self._s3_cache_location.replace("s3://", "").strip("/").split("/", 1)
        bucket = parts[0]
        key = "/".join([parts[1], cache_id]) if len(parts) == 2 else cache_id
        json_str = json.dumps(table, indent=2)
        s3.put_object(Body=json_str, Bucket=bucket, Key=key)

    @staticmethod
    def _cache_id(s3, object_path: str) -> str:
        parts = object_path.replace("s3://", "").split("/", 1)
        response = s3.head_object(Bucket=parts[0], Key=parts[1])
        cache_id = response["ETag"].replace('"', "")
        return cache_id

    def extract_tables(self, document: Document) -> Document:
        s3 = boto3.client("s3")
        cache_id = self._cache_id(s3, document.properties["path"])
        tables = self._get_table(s3, cache_id)

        if tables:
            document.elements = document.elements + tables
            return document

        if tables is not None:
            return document

        # cache miss
        extractor = Textractor(self._profile_name, self._region_name, self._kms_key_id)
        if self._run_full_textract:
            result = extractor.start_document_analysis(
                document.properties["path"],
                TextractFeatures.TABLES,
                s3_upload_path=self._s3_textract_upload_path,
                save_image=False,
            )
            json_data = self._wrap(result, [])
        else:
            # a list holding 1 based page number
            table_pages = [
                element.properties["page_number"] for element in document.elements if element.type == "Table"
            ]
            if not table_pages:
                json_data = {"tables": []}
            else:
                # When no s3 upload path exists, it's assumed textractor won't run even cache miss
                if not self._s3_textract_upload_path:
                    raise RuntimeError("Missing textract upload path")

                # Clip the pages which have tables into a new tmp pdf and upload for textract
                table_pages = list(OrderedDict.fromkeys(table_pages))
                binary = io.BytesIO(document.data["binary_representation"])
                pdf_reader = PyPDF2.PdfReader(binary)
                pdf_writer = PyPDF2.PdfWriter()
                for page_number in table_pages:
                    page = pdf_reader.pages[page_number - 1]  # Page numbers start from 0
                    pdf_writer.add_page(page)
                output_pdf_stream = io.BytesIO()
                pdf_writer.write(output_pdf_stream)

                # Do textract
                result = extractor.start_document_analysis(
                    output_pdf_stream.getvalue(),
                    TextractFeatures.TABLES,
                    s3_upload_path=self._s3_textract_upload_path,
                    save_image=False,
                )
                json_data = self._wrap(result, table_pages)

        # parse table into table element and update cache
        self._put_table(s3, cache_id, json_data)
        document.elements = document.elements + self._parse(json_data)
        return document
