from abc import abstractmethod, ABC
from typing import Optional

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
