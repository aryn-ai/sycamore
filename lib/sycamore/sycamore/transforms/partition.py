from abc import abstractmethod, ABC
import io
from typing import Any, Optional

from bs4 import BeautifulSoup

from ray.data import ActorPoolStrategy

from sycamore.functions import TextOverlapChunker, Chunker
from sycamore.functions import CharacterTokenizer, Tokenizer
from sycamore.functions import reorder_elements
from sycamore.data import BoundingBox, Document, Element, TableElement
from sycamore.plan_nodes import Node
from sycamore.transforms.base import CompositeTransform
from sycamore.transforms.extract_table import TableExtractor
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
from sycamore.utils import use_cuda

from sycamore.transforms.aryn_partitioner import _DEFAULT_ARYN_PARTITIONER_ADDRESS


# This comparator helps sort the elements per page specifically when a page
# has two columns
def _elements_reorder_comparator(element1: Element, element2: Element) -> int:
    # In PixelSpace (default coordinate system), the coordinates of each
    # element starts in the top left corner and proceeds counter-clockwise. The
    # following function checks if the x0 point of the element is in the
    # left column
    def element_in_left_col(e: Element) -> bool:
        if e.bbox is None:
            raise RuntimeError("Element BBox is None")
        return e.bbox.x1 <= 0.5

    page1 = element1.properties["page_number"]
    page2 = element2.properties["page_number"]

    if page1 < page2:
        return -1
    elif page1 > page2:
        return 1
    else:
        if element_in_left_col(element1) and not element_in_left_col(element2):
            return -1
        elif not element_in_left_col(element1) and element_in_left_col(element2):
            return 1
        else:
            return 0


class Partitioner(ABC):
    def __init__(self, device=None, batch_size=1):
        self.device = device
        self.batch_size = batch_size

    @abstractmethod
    def partition(self, document: Document) -> Document:
        pass


class UnstructuredPPTXPartitioner(Partitioner):
    """
    UnstructuredPPTXPartitioner utilizes open-source Unstructured library to extract structured elements from
    unstructured PPTX files.

    Args:
        include_page_breaks: Whether to include page breaks as separate elements.
        strategy: The partitioning strategy to use ("auto" for automatic detection).
        infer_table_structure: Whether to infer table structures in the document.
        ocr_languages: The languages to use for OCR. Default is "eng" (English).
        max_partition_length: The maximum length of each partition (in characters).
        include_metadata: Whether to include metadata in the partitioned elements.

    Example:
         .. code-block:: python

            pptx_partitioner = UnstructuredPPTXPartitioner(
                include_page_breaks=False,
                include_metadata=True,
                include_slide_notes=False,
                chunking_strategy=None,
                **kwargs
            )

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pptx")
                .partition(partitioner=pptx_partitioner)

    """

    @staticmethod
    def to_element(dict: dict[str, Any]) -> Element:
        text = dict.pop("text")
        if isinstance(text, str):
            binary = text.encode("utf-8")
        else:
            binary = text
            text = str(binary, "utf-8")

        element = Element()
        element.type = dict.pop("type", "unknown")
        element.binary_representation = binary
        element.text_representation = text
        element.properties.update(dict.pop("metadata"))
        element.properties.update(dict)

        return element

    def __init__(
        self,
        include_page_breaks: bool = False,
        include_metadata: bool = True,
        include_slide_notes: bool = False,
        chunking_strategy: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device="cpu")
        self._include_page_breaks = include_page_breaks
        self._include_metadata = include_metadata
        self._include_slide_notes = include_slide_notes
        self._chunking_strategy = chunking_strategy
        self._kwargs = kwargs

    def partition(self, document: Document) -> Document:
        from unstructured.partition.pptx import partition_pptx

        binary_file = io.BytesIO(document.data["binary_representation"])

        elements = partition_pptx(
            file=binary_file,
            include_page_breaks=self._include_page_breaks,
            include_metadata=self._include_metadata,
            include_slide_notes=self._include_slide_notes,
            chunking_strategy=self._chunking_strategy,
            **self._kwargs,
        )

        # Here we convert unstructured.io elements into our elements and
        # append them as child elements to the document.
        document.elements = [self.to_element(element.to_dict()) for element in elements]
        del elements

        return document


class UnstructuredPdfPartitioner(Partitioner):
    """
    UnstructuredPdfPartitioner utilizes open-source Unstructured library to extract structured elements from
    unstructured PDFs.

    Args:
        include_page_breaks: Whether to include page breaks as separate elements.
        strategy: The partitioning strategy to use ("auto" for automatic detection).
        infer_table_structure: Whether to infer table structures in the document.
        ocr_languages: The languages to use for OCR. Default is "eng" (English).
        max_partition_length: The maximum length of each partition (in characters).
        include_metadata: Whether to include metadata in the partitioned elements.
        retain_coordinates: Whether to keep the coordinates property from unstructured.
            Default is False. In either case, bbox will be popuplated.

    Example:
         .. code-block:: python

            pdf_partitioner = UnstructuredPdfPartitioner(
                include_page_breaks=True,
                strategy="auto",
                infer_table_structure=True,
                ocr_languages="eng",
                max_partition_length=2000,
                include_metadata=True,
            )

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=pdf_partitioner)

    """

    def __init__(
        self,
        include_page_breaks: bool = False,
        strategy: str = "auto",
        infer_table_structure: bool = False,
        languages: list[str] = ["eng"],
        max_partition_length: Optional[int] = None,
        min_partition_length: Optional[int] = 500,
        include_metadata: bool = True,
        retain_coordinates: bool = False,
    ):
        super().__init__(device="cpu")
        self._include_page_breaks = include_page_breaks
        self._strategy = strategy
        self._infer_table_structure = infer_table_structure
        self._languages = languages
        self._max_partition_length = max_partition_length
        self._min_partition_length = min_partition_length
        self._include_metadata = include_metadata
        self._retain_coordinates = retain_coordinates

    @staticmethod
    def to_element(dict: dict[str, Any], retain_coordinates=False) -> Element:
        text = dict.pop("text")
        if isinstance(text, str):
            binary = text.encode("utf-8")
        else:
            binary = text
            text = str(binary, "utf-8")

        element = Element()
        element.type = dict.pop("type", "unknown")
        element.binary_representation = binary
        element.text_representation = text

        element.properties.update(dict.pop("metadata"))
        element.properties.update(dict)
        coordinates = element.properties.get("coordinates")
        if not retain_coordinates:
            element.properties.pop("coordinates")

        if coordinates is not None:
            x1 = coordinates.get("points")[0][0] / coordinates.get("layout_width")
            y1 = coordinates.get("points")[0][1] / coordinates.get("layout_height")
            x2 = coordinates.get("points")[2][0] / coordinates.get("layout_width")
            y2 = coordinates.get("points")[2][1] / coordinates.get("layout_height")
            element.bbox = BoundingBox(x1, y1, x2, y2)

        return element

    @timetrace("unstructuredPdf")
    def partition(self, document: Document) -> Document:
        from unstructured.partition.pdf import partition_pdf

        binary = io.BytesIO(document.data["binary_representation"])
        try:
            elements = partition_pdf(
                file=binary,
                include_page_breaks=self._include_page_breaks,
                strategy=self._strategy,
                infer_table_structure=self._infer_table_structure,
                languages=self._languages,
                max_partition=self._max_partition_length,
                min_partition=self._min_partition_length,
                include_metadata=self._include_metadata,
            )
        except Exception as e:
            path = document.properties["path"]
            raise RuntimeError(f"UnstructuredPartitioner Error processing {path}") from e

        # Here we convert unstructured.io elements into our elements and
        # set them as the child elements of the document.
        document.elements = [self.to_element(ee.to_dict(), self._retain_coordinates) for ee in elements]
        del elements

        document = reorder_elements(document, _elements_reorder_comparator)
        return document


class HtmlPartitioner(Partitioner):
    """
    HtmlPartitioner processes HTML documents extracting structured content.

    Args:
        skip_headers_and_footers: Whether to skip headers and footers in the document. Default is True.
        extract_tables: Whether to extract tables from the HTML document. Default is False.
        text_chunker: The text chunking strategy to use for processing text content.
        tokenizer: The tokenizer to use for tokenizing text content.

    Example:
         .. code-block:: python

            html_partitioner = HtmlPartitioner(
                skip_headers_and_footers=True,
                extract_tables=True,
                text_chunker=TokenOverlapChunker(chunk_token_count=1000, chunk_overlap_token_count=100),
                tokenizer=CharacterTokenizer(),
            )

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="html")
                .partition(partitioner=html_partitioner)
    """

    def __init__(
        self,
        skip_headers_and_footers: bool = True,
        extract_tables: bool = False,
        text_chunker: Chunker = TextOverlapChunker(),
        tokenizer: Tokenizer = CharacterTokenizer(),
    ):
        super().__init__(device="cpu")
        self._skip_headers_and_footers = skip_headers_and_footers
        self._extract_tables = extract_tables
        self._text_chunker = text_chunker
        self._tokenizer = tokenizer

    @timetrace("beautSoup")
    def partition(self, document: Document) -> Document:
        raw_html = document.binary_representation

        if raw_html is None:
            raise RuntimeError("Attempting to partition invalid document where content=None")

        # note: if content is bytes, BeautifulSoup default to utf-8 encoding
        soup = BeautifulSoup(raw_html, "html.parser")

        # extract title
        titles = soup.find_all("title")
        title = document.doc_id
        if len(titles) > 0:
            title = titles[0].text.replace("\n", "").strip()
        document.properties["title"] = title

        # chunk text and create text elements
        elements = []
        text = soup.get_text(separator=" ", strip=True)
        tokens = self._tokenizer.tokenize(text)
        for chunk in self._text_chunker.chunk(tokens):
            content = "".join(chunk)
            element = Element()
            element.type = "text"
            element.text_representation = content

            element.properties.update(document.properties)
            elements += [element]

        # extract tables
        if self._extract_tables:
            for table in soup.find_all("table"):
                # ignore nested tables
                if len(table.find_all("table")) > 0:
                    continue

                table_element = TableElement()

                # find headers if they exist
                headers = table.findAll("th")
                if len(headers) > 0:
                    table_element.columns = [tag.text for tag in headers]

                table_element.text_representation = table.text
                table_element.properties.update(document.properties)

                # parse all rows, use all text as content
                rows = table.findAll("tr")
                table_element.rows = []
                for row in rows:
                    cols = row.findAll("td")
                    if len(cols) > 0:
                        row_vals = [tag.text for tag in cols]
                        table_element.rows += [row_vals]
                elements.append(table_element)
        document.elements = document.elements + elements

        return document


SYCAMORE_DETR_MODEL = "Aryn/deformable-detr-DocLayNet"


class SycamorePartitioner(Partitioner):
    """
    The SycamorePartitioner uses an object recognition model to partition the document into
    structured elements.

    Args:
        model_name_or_path: The HuggingFace coordinates or model local path. Should be set to
             the default SYCAMORE_DETR_MODEL unless you are testing a custom model. 
        threshold: The threshold to use for accepting the models predicted bounding boxes. A lower
             value will include more objects, but may have overlaps, a higher value will reduce the
             number of overlaps, but may miss legitimate objects. 
        use_ocr: Whether to use OCR to extract text from the PDF. If false, we will attempt to extract
             the text from the underlying PDF. 
        ocr_images: If set with use_ocr, will attempt to OCR regions of the document identified as images. 
        ocr_tables: If set with use_ocr, will attempt to OCR regions on the document identified as tables.
             Should not be set when `extract_table_structure` is true. 
        extract_table_structure: If true, runs a separate table extraction model to extract cells from
             regions of the document identified as tables. 
        table_structure_extractor: The table extraction implementaion to use when extract_table_structure
             is True. The default is the TableTransformerStructureExtractor. 
        extract_images: If true, crops each region identified as an image and attaches it to the associated
             ImageElement. This can later be fed into the SummarizeImages transform.
    
    Example:
         The following shows an example of using the SycamorePartitioner to partition a PDF and extract
         both table structure and image
    
         .. code-block:: python

            context = scyamore.init()
            partitioner = SycamorePartitioner(extract_table_structure=True, extract_images=True)
            context.read.binary(paths, binary_format="pdf")\
                 .partition(partitioner=partitioner)
    """

    def __init__(
        self,
        model_name_or_path=SYCAMORE_DETR_MODEL,
        threshold: float = 0.4,
        use_ocr=False,
        ocr_images=False,
        ocr_tables=False,
        extract_table_structure=False,
        table_structure_extractor=None,
        extract_images=False,
        device=None,
        model_server_endpoint=None,
        batch_size: int = 1,
    ):
        if not device:
            device = "cuda" if use_cuda() else "cpu"
        super().__init__(device=device, batch_size=batch_size)
        self._model_name_or_path = model_name_or_path
        self._device = device
        self._threshold = threshold
        self._use_ocr = use_ocr
        self._ocr_images = ocr_images
        self._ocr_tables = ocr_tables
        self._extract_table_structure = extract_table_structure
        self._table_structure_extractor = table_structure_extractor
        self._extract_images = extract_images
        self._model_server_endpoint = model_server_endpoint
        self._batch_size = batch_size

    # For now, we reorder elements based on page, left/right column, y axle position then finally x axle position
    @staticmethod
    def _elements_reorder(element1: Element, element2: Element) -> int:
        def element_in_left_col(e: Element) -> bool:
            if e.bbox is None:
                raise RuntimeError("Element BBox is None")
            return e.bbox.x1 <= 0.5

        page1 = element1.properties["page_number"]
        page2 = element2.properties["page_number"]
        bbox1 = element1.bbox
        bbox2 = element2.bbox

        if page1 < page2:
            return -1
        elif page1 > page2:
            return 1
        elif element_in_left_col(element1) and not element_in_left_col(element2):
            return -1
        elif not element_in_left_col(element1) and element_in_left_col(element2):
            return 1
        elif bbox1 is None or bbox2 is None:
            return 0
        elif bbox1.y1 < bbox2.y1:
            return -1
        elif bbox1.y1 > bbox2.y1:
            return 1
        elif bbox1.x1 < bbox2.x1:
            return -1
        elif bbox1.x1 > bbox2.x1:
            return 1
        else:
            return 0

    @timetrace("SycamorePdf")
    def partition(self, document: Document) -> Document:
        binary = io.BytesIO(document.data["binary_representation"])
        from sycamore.transforms.detr_partitioner import SycamorePDFPartitioner

        partitioner = SycamorePDFPartitioner(self._model_name_or_path, device=self._device)

        try:
            result = partitioner.partition_pdf(
                binary,
                self._threshold,
                use_ocr=self._use_ocr,
                ocr_images=self._ocr_images,
                ocr_tables=self._ocr_tables,
                extract_table_structure=self._extract_table_structure,
                table_structure_extractor=self._table_structure_extractor,
                extract_images=self._extract_images,
                model_server_endpoint=self._model_server_endpoint,
                batch_size=self._batch_size,
            )
        except Exception as e:
            path = document.properties["path"]
            raise RuntimeError(f"SycamorePartitioner Error processing {path}") from e

        elements = []
        for i, r in enumerate(result):
            for ele in r:
                ele.properties["page_number"] = i + 1
                elements.append(ele)

        document.elements = elements
        document = reorder_elements(document, self._elements_reorder)
        return document


class ArynPartitioner(Partitioner):
    """
    The ArynPartitioner runs a SycamorePartitioner on Aryn's GPU servers to offload compute.

    Args:
        aryn_token: The account token used to authenticate with Aryn's servers.
        threshold: The threshold to use for accepting the models predicted bounding boxes. A lower
             value will include more objects, but may have overlaps, a higher value will reduce the
             number of overlaps, but may miss legitimate objects.
        use_ocr: Whether to use OCR to extract text from the PDF. If false, we will attempt to extract
             the text from the underlying PDF.
        ocr_images: If set with use_ocr, will attempt to OCR regions of the document identified as images.
        ocr_tables: If set with use_ocr, will attempt to OCR regions on the document identified as tables.
             Should not be set when `extract_table_structure` is true.
        extract_table_structure: If true, runs a separate table extraction model to extract cells from
             regions of the document identified as tables.
        extract_images: If true, crops each region identified as an image and attaches it to the associated
             ImageElement. This can later be fed into the SummarizeImages transform.

    Example:
        The following shows an example of using the ArynPartitioner to partition a PDF and extract
        both table structure and images.

        .. code-block:: python

            context = sycamore.init()
            partitioner = ArynPartitioner(extract_table_structure=True, extract_images=True)
            context.read.binary(paths, binary_format="pdf")\
                .partition(partitioner=partitioner)
    """

    def __init__(
        self,
        aryn_token: str,
        threshold: float = 0.4,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_tables: bool = False,
        extract_table_structure: bool = False,
        extract_images: bool = False,
        aryn_partitioner_address: str = _DEFAULT_ARYN_PARTITIONER_ADDRESS,
    ):
        super().__init__(device="cpu", batch_size=1)
        self._aryn_token = aryn_token
        self._threshold = threshold
        self._use_ocr = use_ocr
        self._ocr_images = ocr_images
        self._ocr_tables = ocr_tables
        self._extract_table_structure = extract_table_structure
        self._extract_images = extract_images
        self._aryn_partitioner_address = aryn_partitioner_address

    def partition(self, document: Document):
        binary = io.BytesIO(document.data["binary_representation"])
        from sycamore.transforms.aryn_partitioner import ArynPDFPartitioner

        try:
            result = ArynPDFPartitioner.partition_pdf(
                binary,
                self._aryn_token,
                threshold=self._threshold,
                use_ocr=self._use_ocr,
                ocr_images=self._ocr_images,
                ocr_tables=self._ocr_tables,
                extract_table_structure=self._extract_table_structure,
                extract_images=self._extract_images,
            )
        except Exception as e:
            path = document.properties["path"]
            raise RuntimeError(f"ArynPartitioner Error Processing {path}") from e

        document.elements = result
        document = reorder_elements(document, SycamorePartitioner._elements_reorder)
        return document


class Partition(CompositeTransform):
    """
    The Partition transform segments documents into elements. For example, a typical partitioner might chunk a document
    into elements corresponding to paragraphs, images, and tables. Partitioners are format specific, so for instance for
    HTML you can use the HtmlPartitioner and for PDFs, we provide the UnstructuredPdfPartitioner, which utilizes the
    unstructured open-source library.

    Args:
        child: The source node or component that provides the dataset to be embedded.
        partitioner: An instance of a Partitioner class to be applied
        resource_args: Additional resource-related arguments that can be passed to the Partition operation.

    Example:
         .. code-block:: python

            source_node = ...  # Define a source node or component that provides a dataset.
            custom_partitioner = MyPartitioner(partitioner_params)
            partition_transform = Partition(child=source_node, partitioner=custom_partitioner)
            partitioned_dataset = partition_transform.execute()
    """

    def __init__(
        self, child: Node, partitioner: Partitioner, table_extractor: Optional[TableExtractor] = None, **resource_args
    ):
        ops = []

        if partitioner.device == "cuda":
            if "num_gpus" not in resource_args:
                resource_args["num_gpus"] = 1.0
            assert resource_args["num_gpus"] >= 0
            if "compute" not in resource_args:
                resource_args["compute"] = ActorPoolStrategy(size=1)
            assert isinstance(resource_args["compute"], ActorPoolStrategy)
            if "batch_size" not in resource_args:
                resource_args["batch_size"] = partitioner.batch_size
        elif partitioner.device == "cpu":
            resource_args.pop("num_gpus", None)

        ops = [{**resource_args, "f": Map.wrap(partitioner.partition)}]
        if table_extractor is not None:
            ops.append({"f": Map.wrap(table_extractor.extract_tables)})

        # Note: we are not applying resource args to the entire composite operation just the first step because that
        # matches with the original code. It is unclear if this is the correct behavior.
        super().__init__(child, ops)
