import io
import json
import logging
from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import Optional

import PyPDF2
import boto3
from botocore.exceptions import ClientError
from textractor import Textractor
from textractor.data.constants import TextractFeatures
from textractor.entities.document import Document as TextractorDocument
from textractor.entities.lazy_document import LazyDocument as LazyTextractorDocument
from textractor.parsers import response_parser

from sycamore.data import BoundingBox, Document, Element

logger = logging.getLogger("sycamore")


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

    def get_textract_result(self, document: Document) -> Optional[TextractorDocument | LazyTextractorDocument]:

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
                document.properties["path"], TextractFeatures.TABLES, s3_upload_path=dest)
        return result

    @staticmethod
    def get_tables_from_textract_result(result: TextractorDocument | LazyTextractorDocument):
        # https://docs.aws.amazon.com/textract/latest/dg/API_BoundingBox.html
        def bbox_to_coord(bbox):
            return bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height

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
        textract_result = self.get_textract_result(document)
        tables = self.get_tables_from_textract_result(textract_result)
        document.elements = document.elements + tables
        return document


class CachedTextractTableExtractor(TextractTableExtractor):
    """
    Extends TextractTableExtractor with S3 based cache support for raw Textract results.

    CachedTextractTableExtractor overrides the 'get_textract_result' method by doing the following:
     1. if cache hit for current document, get from cache and return, otherwise continue
     2. if run_full_textract is enabled, call textractor on the whole document and go to step 4
     3. else clip pages which contain tables and run table extraction using textractor
     5. update cache accordingly based on textractor result and return result
    """

    def __init__(self, s3_cache_location, run_full_textract: bool = False, s3_textract_upload_path: str = "",
                 profile_name: Optional[str] = None, region_name: Optional[str] = None, kms_key_id: str = ""):
        super().__init__(profile_name, region_name, kms_key_id)
        self._s3_cache_location = s3_cache_location
        self._run_full_textract = run_full_textract
        self._s3_textract_upload_path = s3_textract_upload_path
        self._profile_name = profile_name
        self._region_name = region_name
        self._kms_key_id = kms_key_id

    def _get_cached_textract_result(self, s3, cache_id: str) -> [Optional[TextractorDocument], dict]:
        """Get cache from S3"""
        try:
            parts = self._s3_cache_location.replace("s3://", "").strip("/").split("/", 1)
            bucket = parts[0]
            key = "/".join([parts[1], cache_id]) if len(parts) == 2 else cache_id
            response = s3.get_object(Bucket=bucket, Key=key)
            parsed_response = json.loads(response['Body'].read())
            return response_parser.parse(parsed_response["textract_result"]), parsed_response["document_page_mapping"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None, None
            else:
                raise

    def _cache_textract_result(self, s3, cache_id: str,
                               result: TextractorDocument | LazyTextractorDocument,
                               document_page_mapping: list):
        """Put table into S3"""
        parts = self._s3_cache_location.replace("s3://", "").strip("/").split("/", 1)
        bucket = parts[0]
        key = "/".join([parts[1], cache_id]) if len(parts) == 2 else cache_id
        json_str = json.dumps({
            "document_page_mapping": document_page_mapping,
            "textract_result": result.response
        })
        s3.put_object(Body=json_str, Bucket=bucket, Key=key)

    @staticmethod
    def _cache_id(s3, object_path: str) -> str:
        parts = object_path.replace("s3://", "").split("/", 1)
        response = s3.head_object(Bucket=parts[0], Key=parts[1])
        cache_id = response["ETag"].replace('"', "")
        return cache_id

    def get_textract_result(self, document: Document) -> [Optional[TextractorDocument | LazyTextractorDocument], dict]:
        s3 = boto3.client("s3")
        cache_id = self._cache_id(s3, document.properties["path"])
        try:
            textract_result, document_page_mapping = self._get_cached_textract_result(s3, cache_id)
            if textract_result:
                logger.info(f"Textract cache hit for {document.properties['path']}")
                return textract_result, document_page_mapping
        except Exception as e:
            logger.exception("Error in reading from cache %s", str(e))

        # cache miss

        document_page_mapping = list(OrderedDict.fromkeys(
            [element.properties["page_number"] for element in document.elements if element.type == "Table"]
        ))

        # no pages with tables found and no full execution
        if not self._run_full_textract and not document_page_mapping:
            return None, None
        logger.info(f"Textract cache miss for {document.properties['path']}")
        extractor = Textractor(self._profile_name, self._region_name, self._kms_key_id)
        if self._run_full_textract:
            textract_result = extractor.start_document_analysis(
                document.properties["path"],
                TextractFeatures.TABLES,
                s3_upload_path=self._s3_textract_upload_path,
                save_image=False,
            )
        else:
            # When no s3 upload path exists, it's assumed textractor won't run even cache miss
            if not self._s3_textract_upload_path:
                raise RuntimeError("Missing textract upload path")

            # Clip the pages which have tables into a new tmp pdf and upload for textract
            binary = io.BytesIO(document.data["binary_representation"])
            pdf_reader = PyPDF2.PdfReader(binary)
            pdf_writer = PyPDF2.PdfWriter()
            for page_number in document_page_mapping:
                page = pdf_reader.pages[page_number - 1]  # Page numbers start from 0
                pdf_writer.add_page(page)
            output_pdf_stream = io.BytesIO()
            pdf_writer.write(output_pdf_stream)

            # Do textract
            textract_result = extractor.start_document_analysis(
                output_pdf_stream.getvalue(),
                TextractFeatures.TABLES,
                s3_upload_path=self._s3_textract_upload_path,
                save_image=False,
            )

        self._cache_textract_result(s3, cache_id, textract_result, document_page_mapping)

        return textract_result, document_page_mapping

    def extract_tables(self, document: Document) -> Document:
        textract_result, document_page_mapping = self.get_textract_result(document)
        if textract_result:
            tables = self.get_tables_from_textract_result(textract_result)
            # put back actual page numbers
            for table in tables:
                table.properties["page_number"] = document_page_mapping[table.properties["page_number"] - 1] \
                    if document_page_mapping else table.properties["page_number"]
            document.elements = document.elements + tables
        return document
