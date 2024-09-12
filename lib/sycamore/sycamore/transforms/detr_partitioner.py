import gc
import logging
import os
import tempfile
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from io import IOBase
from typing import cast, Any, BinaryIO, List, Optional, Union
from pathlib import Path
import pwd

import requests
import json
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_delay
import base64
import pdf2image
from PIL import Image
import fasteners
from pypdf import PdfReader

from sycamore.data import Element, BoundingBox, ImageElement, TableElement
from sycamore.data.element import create_element
from sycamore.transforms.table_structure.extract import DEFAULT_TABLE_STRUCTURE_EXTRACTOR
from sycamore.utils import choose_device
from sycamore.utils.cache import Cache
from sycamore.utils.image_utils import crop_to_bbox, image_to_bytes
from sycamore.utils.import_utils import requires_modules
from sycamore.utils.memory_debugging import display_top, gc_tensor_dump
from sycamore.utils.pdf import convert_from_path_streamed_batched
from sycamore.utils.time_trace import LogTime, timetrace
from sycamore.transforms.text_extraction import TextExtractor, OCRModel, EXTRACTOR_DICT

logger = logging.getLogger(__name__)
_DETR_LOCK_FILE = f"{pwd.getpwuid(os.getuid()).pw_dir}/.cache/Aryn-Detr.lock"
_VERSION = "0.2024.07.24"

def _batchify(iterable, n=1):
    length = len(iterable)
    for i in range(0, length, n):
        yield iterable[i : min(i + n, length)]


ARYN_DETR_MODEL = "Aryn/deformable-detr-DocLayNet"
DEFAULT_ARYN_PARTITIONER_ADDRESS = "https://api.aryn.cloud/v1/document/partition"
_TEN_MINUTES = 600


class ArynPDFPartitionerException(Exception):
    def __init__(self, message, can_retry=False):
        super().__init__(message)
        self.can_retry = can_retry


def _can_retry(e: BaseException) -> bool:
    if isinstance(e, ArynPDFPartitionerException):
        return e.can_retry
    else:
        return False


def get_page_count(fp: BinaryIO):
    fp.seek(0)
    reader = PdfReader(fp)
    num_pages = len(reader.pages)
    fp.seek(0)
    return num_pages


class ArynPDFPartitioner:
    """
    This class contains the implementation of PDF partitioning using a Deformable DETR model.

    This is an implementation class. Callers looking to partition a DocSet should use the
    ArynPartitioner class.
    """

    def __init__(self, model_name_or_path=ARYN_DETR_MODEL, device=None, cache: Optional[Cache] = None):
        """
        Initializes the ArynPDFPartitioner and underlying DETR model.

        Args:
            model_name_or_path: The HuggingFace coordinates or local path to the DeformableDETR weights to use.
            device: The device on which to run the model.
        """
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.device = device
        self.cache = cache

    def _init_model(self):
        if self.model is None:
            assert self.model_name_or_path is not None
            with LogTime("init_detr_model"):
                self.model = DeformableDetr(self.model_name_or_path, self.device, self.cache)

    @staticmethod
    def _supplement_text(inferred: List[Element], text: List[Element], threshold: float = 0.5) -> List[Element]:
        # We first check IOU between inferred object and pdf miner text object, we also check if a detected object
        # fully contains a pdf miner text object. After that, we combined all texts belonging a detected object and
        # update its text representation. We allow multiple detected objects contain the same text, we hold on solving
        # this.

        unmatched = text.copy()
        for index_i, i in enumerate(inferred):
            matched = []
            for t in text:
                if i.bbox and t.bbox and (i.bbox.iou(t.bbox) > threshold or i.bbox.contains(t.bbox)):
                    matched.append(t)
                    if t in unmatched:
                        unmatched.remove(t)
            if matched:
                matches = []
                full_text = []
                for m in matched:
                    matches.append(m)
                    if m.text_representation:
                        full_text.append(m.text_representation)

                if isinstance(i, TableElement):
                    i.tokens = [{"text": elem.text_representation, "bbox": elem.bbox} for elem in matches]

                i.data["text_representation"] = " ".join(full_text)

        return inferred + unmatched

    def partition_pdf(
        self,
        file: BinaryIO,
        threshold: float = 0.4,
        use_ocr=False,
        ocr_images=False,
        ocr_model="easyocr",
        per_element_ocr=False,
        extract_table_structure=False,
        table_structure_extractor=None,
        extract_images=False,
        batch_size: int = 1,
        batch_at_a_time=True,
        use_partitioning_service=True,
        aryn_api_key: str = "",
        aryn_partitioner_address=DEFAULT_ARYN_PARTITIONER_ADDRESS,
        use_cache=False,
        pages_per_call: int = -1,
    ) -> List[Element]:
        if use_partitioning_service:
            assert aryn_api_key != ""
            return self._partition_remote(
                file=file,
                aryn_api_key=aryn_api_key,
                aryn_partitioner_address=aryn_partitioner_address,
                threshold=threshold,
                use_ocr=use_ocr,
                ocr_images=ocr_images,
                ocr_model=ocr_model,
                per_element_ocr=per_element_ocr,
                extract_table_structure=extract_table_structure,
                extract_images=extract_images,
                pages_per_call=pages_per_call,
            )
        else:
            if batch_at_a_time:
                temp = self._partition_pdf_batched(
                    file=file,
                    threshold=threshold,
                    use_ocr=use_ocr,
                    ocr_images=ocr_images,
                    ocr_model=ocr_model,
                    per_element_ocr=per_element_ocr,
                    extract_table_structure=extract_table_structure,
                    table_structure_extractor=table_structure_extractor,
                    extract_images=extract_images,
                    batch_size=batch_size,
                    use_cache=use_cache,
                )
            else:
                temp = self._partition_pdf_sequenced(
                    file=file,
                    threshold=threshold,
                    use_ocr=use_ocr,
                    ocr_images=ocr_images,
                    ocr_model=ocr_model,
                    per_element_ocr=per_element_ocr,
                    extract_table_structure=extract_table_structure,
                    table_structure_extractor=table_structure_extractor,
                    extract_images=extract_images,
                    batch_size=batch_size,
                    use_cache=use_cache,
                )
            elements = []
            for i, r in enumerate(temp):
                for ele in r:
                    ele.properties["page_number"] = i + 1
                    elements.append(ele)
            return elements

    @staticmethod
    @retry(
        retry=retry_if_exception(_can_retry),
        wait=wait_exponential(multiplier=1, min=1),
        stop=stop_after_delay(_TEN_MINUTES),
    )
    def _call_remote_partitioner(
        file: BinaryIO,
        aryn_api_key: str,
        aryn_partitioner_address=DEFAULT_ARYN_PARTITIONER_ADDRESS,
        threshold: float = 0.4,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_model: str = "easyocr",
        per_element_ocr: bool = False,
        extract_table_structure: bool = False,
        extract_images: bool = False,
        selected_pages: list = [],
    ) -> List[Element]:
        file.seek(0)
        options = {
            "threshold": threshold,
            "use_ocr": use_ocr,
            "ocr_images": ocr_images,
            "ocr_model": ocr_model,
            "per_element_ocr": per_element_ocr,
            "extract_table_structure": extract_table_structure,
            "extract_images": extract_images,
            "selected_pages": selected_pages,
            "source": "sycamore",
        }

        files: Mapping = {"pdf": file, "options": json.dumps(options).encode("utf-8")}
        header = {"Authorization": f"Bearer {aryn_api_key}"}

        logger.debug(f"ArynPartitioner POSTing to {aryn_partitioner_address} with files={files}")
        response = requests.post(aryn_partitioner_address, files=files, headers=header, stream=True)
        content = []
        in_status = False
        in_bulk = False
        partial_line = b""
        for part in response.iter_content(None):
            if not part:
                continue

            content.append(part)
            if in_bulk:
                continue
            partial_line = partial_line + part
            if b"\n" not in part:
                # Make sure we don't go O(n^2) from constantly appending to our partial_line.
                if len(partial_line) > 100000:
                    logger.warning("Too many bytes without newline. Skipping incremental status")
                    in_bulk = True

                continue

            lines = partial_line.split(b"\n")
            if part.endswith(b"\n"):
                partial_line = b""
            else:
                partial_line = lines.pop()

            for line in lines:
                if line.startswith(b'  "status"'):
                    in_status = True
                if not in_status:
                    continue
                if line.startswith(b"  ],"):
                    in_status = False
                    in_bulk = True
                    continue
                if line.startswith(b'    "T+'):
                    t = json.loads(line.decode("utf-8").removesuffix(","))
                    logger.info(f"ArynPartitioner: {t}")

        body = b"".join(content).decode("utf-8")
        logger.debug("ArynPartitioner Recieved data")

        if response.status_code != 200:
            if response.status_code == 500 or response.status_code == 502:
                logger.debug(
                    "ArynPartitioner recieved a retry-able error {} x-aryn-call-id: {}".format(
                        response, response.headers.get("x-aryn-call-id")
                    )
                )
                raise ArynPDFPartitionerException(
                    "Error: status_code: {}, reason: {} (x-aryn-call-id: {})".format(
                        response.status_code, body, response.headers.get("x-aryn-call-id")
                    ),
                    can_retry=True,
                )
            raise ArynPDFPartitionerException(
                "Error: status_code: {}, reason: {} (x-aryn-call-id: {})".format(
                    response.status_code, body, response.headers.get("x-aryn-call-id")
                )
            )

        response_json = json.loads(body)
        if isinstance(response_json, dict):
            status = response_json.get("status", [])
            if "error" in response_json:
                raise ArynPDFPartitionerException(
                    f"Error partway through processing: {response_json['error']}\nPartial Status:\n{status}"
                )
            response_json = response_json.get("elements", [])

        elements = []
        for element_json in response_json:
            element = create_element(**element_json)
            if element.binary_representation:
                element.binary_representation = base64.b64decode(element.binary_representation)
            elements.append(element)

        return elements

    @staticmethod
    def _partition_remote(
        file: BinaryIO,
        aryn_api_key: str,
        aryn_partitioner_address=DEFAULT_ARYN_PARTITIONER_ADDRESS,
        threshold: float = 0.4,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_model: str = "easyocr",
        per_element_ocr: bool = False,
        extract_table_structure: bool = False,
        extract_images: bool = False,
        pages_per_call: int = -1,
    ) -> List[Element]:
        page_count = get_page_count(file)

        result: List[Element] = []
        low = 1
        high = pages_per_call
        if pages_per_call == -1:
            high = page_count
        while low <= page_count:
            result.extend(
                ArynPDFPartitioner._call_remote_partitioner(
                    file=file,
                    aryn_api_key=aryn_api_key,
                    aryn_partitioner_address=aryn_partitioner_address,
                    threshold=threshold,
                    use_ocr=use_ocr,
                    ocr_images=ocr_images,
                    ocr_model=ocr_model,
                    per_element_ocr=per_element_ocr,
                    extract_table_structure=extract_table_structure,
                    extract_images=extract_images,
                    selected_pages=[[low, min(high, page_count)]],
                )
            )
            low = high + 1
            high += pages_per_call

        return result

    def _partition_pdf_sequenced(
        self,
        file: BinaryIO,
        threshold: float = 0.4,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_model: str = "easyocr",
        per_element_ocr: bool = False,
        extract_table_structure: bool = False,
        table_structure_extractor=None,
        extract_images: bool = False,
        batch_size: int = 1,
        use_cache=False,
    ) -> List[List["Element"]]:
        """
        Partitions a PDF with the DeformableDETR model.

        Args:
           file: A file-like object containing the PDF. Generally this is a wrapper around binary_representation.
           threshold: The threshold to use for accepting the model's predicted bounding boxes.
           use_ocr: Whether to use OCR to extract text from the PDF
           ocr_images: If set with use_ocr, will attempt to OCR regions of the document identified as images.
           ocr_model: If set with use_ocr, will use the model specified by this argument. Valid options are "easyocr",
                "tesseract", "paddle", and "legacy". If you choose paddle make sure to install paddlepaddle
                or paddlepaddle-gpu if you have a CPU or GPU. Further details are found below:
                https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html
           per_element_ocr= If set with use_ocr, will execute OCR on each element rather than the entire page.
           extract_table_structure: If true, runs a separate table extraction model to extract cells from
             regions of the document identified as tables.
           table_structure_extractor: The table extraction implementaion to use when extract_table_structure is True.
           extract_images: If true, crops each region identified as an image and
             attaches it to the associated ImageElement.

        Returns:
           A list of lists of Elements. Each sublist corresponds to a page in the original PDF.
        """
        self._init_model()

        if not table_structure_extractor:
            table_structure_extractor = DEFAULT_TABLE_STRUCTURE_EXTRACTOR(device=self.device)

        LogTime("partition_start", point=True)
        with LogTime("convert2bytes"):
            images: list[Image.Image] = pdf2image.convert_from_bytes(file.read())

        with LogTime("toRGB"):
            images = [im.convert("RGB") for im in images]

        batches = _batchify(images, batch_size)
        deformable_layout: list[list[Element]] = []
        with LogTime("all_batches"):
            for i, batch in enumerate(batches):
                with LogTime(f"infer_one_batch {i}/{len(images) / batch_size}"):
                    assert self.model is not None
                    deformable_layout += self.model.infer(batch, threshold, use_cache)
        # The cast here is to make mypy happy. PDFMiner expects IOBase,
        # but typing.BinaryIO doesn't extend from it. BytesIO
        # (the concrete class) implements both.
        file_name = cast(IOBase, file)
        hash_key = Cache.get_hash_context(file_name.read()).hexdigest()
        if not use_ocr or not per_element_ocr:
            with LogTime("text_extract", log_start=True):
                extracted_layout = self._run_text_extractor(
                    use_ocr=use_ocr,
                    ocr_model=ocr_model,
                    ocr_images=ocr_images,
                    file_name=file_name,
                    hash_key=hash_key,
                    use_cache=use_cache,
                    images=images,
                )
            # page count should be the same
            assert len(extracted_layout) == len(deformable_layout)
            with LogTime("text_supplement"):
                for d, p in zip(deformable_layout, extracted_layout):
                    self._supplement_text(d, p)
        else:
            extract_ocr(images, deformable_layout, ocr_images=ocr_images, ocr_model_name=ocr_model)
        if extract_table_structure or extract_images:
            with LogTime("extract_images_or_table"):
                for i, page_elements in enumerate(deformable_layout):
                    with LogTime(f"extract_images_or_table_one {i}/{len(deformable_layout)}"):
                        image = images[i]
                        for element in page_elements:
                            if isinstance(element, TableElement) and extract_table_structure:
                                table_structure_extractor.extract(element, image)

                            if isinstance(element, ImageElement) and extract_images:
                                if element.bbox is None:
                                    continue
                                cropped_image = crop_to_bbox(image, element.bbox).convert("RGB")
                                element.binary_representation = image_to_bytes(cropped_image)
                                element.image_mode = cropped_image.mode
                                element.image_size = cropped_image.size
                                # print(element.properties)
        LogTime("finish", point=True)
        return deformable_layout

    def _partition_pdf_batched(
        self,
        file: BinaryIO,
        threshold: float = 0.4,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_model: str = "easyocr",
        per_element_ocr: bool = False,
        extract_table_structure: bool = False,
        table_structure_extractor=None,
        extract_images: bool = False,
        batch_size: int = 1,
        use_cache=False,
    ) -> List[List["Element"]]:
        self._init_model()

        LogTime("partition_start", point=True)
        with tempfile.NamedTemporaryFile(prefix="detr-pdf-input-") as pdffile:
            with LogTime("write_pdf"):
                file_hash = Cache.get_hash_context_file(pdffile.name)
                data = file.read()
                data_len = len(data)
                pdffile.write(data)
                del data
                pdffile.flush()
                logger.info(f"Wrote {pdffile.name}")
            stat = os.stat(pdffile.name)
            assert stat.st_size == data_len
            return self._partition_pdf_batched_named(
                pdffile.name,
                file_hash.hexdigest(),
                threshold,
                use_ocr,
                ocr_images,
                ocr_model,
                per_element_ocr,
                extract_table_structure,
                table_structure_extractor,
                extract_images,
                batch_size,
                use_cache,
            )

    def _partition_pdf_batched_named(
        self,
        filename: str,
        hash_key: str,
        threshold: float = 0.4,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_model: str = "easyocr",
        per_element_ocr: bool = False,
        extract_table_structure=False,
        table_structure_extractor=None,
        extract_images=False,
        batch_size: int = 1,
        use_cache=False,
    ) -> List[List["Element"]]:
        self._init_model()

        if extract_table_structure and not table_structure_extractor:
            table_structure_extractor = DEFAULT_TABLE_STRUCTURE_EXTRACTOR(device=self.device)

        text_extractor = None
        exec = ProcessPoolExecutor(max_workers=1)
        logging.getLogger("easyocr").setLevel(logging.DEBUG)
        if not use_ocr or not per_element_ocr:
            with LogTime("start_text_extractor", log_start=True):
                print("start_text_extractor_print")
                text_extractor = exec.submit(
                    self._run_text_extractor,
                    use_ocr=use_ocr,
                    ocr_model=ocr_model,
                    ocr_images=ocr_images,
                    file_name=filename,
                    hash_key=hash_key,
                    use_cache=use_cache,
                )

        deformable_layout = []
        if tracemalloc.is_tracing():
            before = tracemalloc.take_snapshot()
        for i in convert_from_path_streamed_batched(filename, batch_size):
            print("process_batch_inference_print")
            parts = self.process_batch_inference(
                i,
                threshold=threshold,
                use_cache=use_cache,
                use_ocr=use_ocr,
                ocr_model=ocr_model,
                ocr_images=ocr_images,
                per_element_ocr=per_element_ocr,
            )
            assert len(parts) == len(i)
            deformable_layout.extend(parts)
            if tracemalloc.is_tracing():
                gc.collect()
                after = tracemalloc.take_snapshot()
                top_stats = after.compare_to(before, "lineno")

                print("[ Top 10 differences ]")
                for stat in top_stats[:10]:
                    print(stat)
                before = after
                display_top(after)
        print("text_extractor_print")
        if text_extractor is not None:
            with LogTime("wait_for_text_extractor", log_start=True):
                text_extractor_layout = text_extractor.result(timeout=10)
            assert len(text_extractor_layout) == len(
                deformable_layout
            ), f"{len(text_extractor_layout)} vs {len(deformable_layout)}"
            with LogTime("text_extractor_supplement"):
                for d, p in zip(deformable_layout, text_extractor_layout):
                    self._supplement_text(d, p)
        # TODO: optimize this to make pdfminer also streamed so we can process each page in sequence without
        # having to double-convert the document
        for i in convert_from_path_streamed_batched(filename, batch_size):
            self.process_batch_extraction(
                i,
                deformable_layout,
                extract_table_structure=extract_table_structure,
                table_structure_extractor=table_structure_extractor,
                extract_images=extract_images,
            )
            assert len(parts) == len(i)
        if tracemalloc.is_tracing():
            (current, peak) = tracemalloc.get_traced_memory()
            logger.info(f"Memory Usage current={current} peak={peak}")
            top = tracemalloc.take_snapshot()
            display_top(top)
        return deformable_layout

    @staticmethod
    def _run_text_extractor(
        use_ocr: bool,
        ocr_images: bool,
        file_name: Union[str, IOBase],
        hash_key: str,
        use_cache: bool,
        ocr_model: str,
        images: Optional[list[Image.Image]] = None,
    ):
        logging.getLogger("easyocr").setLevel(logging.DEBUG)
        print("start_text_extractor_print_2")
        kwargs = {"ocr_images": ocr_images, "images": images}
        if not use_ocr:
            ocr_model = "pdfminer"
        if ocr_model not in EXTRACTOR_DICT.keys():
            raise ValueError(f"Unknown ocr_model: {ocr_model}")
        model: TextExtractor = EXTRACTOR_DICT[ocr_model]()
        with LogTime("text_extract", log_start=True):
            extracted_layout = model.extract(file_name, hash_key, use_cache, **kwargs)
        print("end_text_extractor_print")
        return extracted_layout

    def process_batch_inference(
        self,
        batch: list[Image.Image],
        threshold: float,
        use_cache: bool,
        use_ocr: bool,
        ocr_model: str,
        ocr_images: bool,
        per_element_ocr: bool,
    ) -> Any:
        self._init_model()

        with LogTime("infer"):
            assert self.model is not None
            deformable_layout = self.model.infer(batch, threshold, use_cache)

        gc_tensor_dump()
        assert len(deformable_layout) == len(batch)
        if use_ocr and per_element_ocr:
            extract_ocr(
                batch,
                deformable_layout,
                ocr_images=ocr_images,
                ocr_model_name=ocr_model,
            )
        # else pdfminer happens in parent since it is whole document.
        return deformable_layout

    def process_batch_extraction(
        self,
        batch: list[Image.Image],
        deformable_layout: Any,
        extract_table_structure,
        table_structure_extractor,
        extract_images,
    ) -> Any:

        if extract_table_structure:
            with LogTime("extract_table_structure_batch"):
                if table_structure_extractor is None:
                    table_structure_extractor = DEFAULT_TABLE_STRUCTURE_EXTRACTOR(device=self.device)
                for i, page_elements in enumerate(deformable_layout):
                    image = batch[i]
                    for element in page_elements:
                        if isinstance(element, TableElement):
                            table_structure_extractor.extract(element, image)

        if extract_images:
            with LogTime("extract_images_batch"):
                for i, page_elements in enumerate(deformable_layout):
                    image = batch[i]
                    for element in page_elements:
                        if isinstance(element, ImageElement) and element.bbox is not None:
                            cropped_image = crop_to_bbox(image, element.bbox).convert("RGB")
                            element.binary_representation = image_to_bytes(cropped_image)
                            element.image_mode = cropped_image.mode
                            element.image_size = cropped_image.size

        return deformable_layout


class SycamoreObjectDetection(ABC):
    """Wrapper class for the various object detection models."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def infer(self, image: List[Image.Image], threshold: float) -> List[List[Element]]:
        """Do inference using the wrapped model."""
        pass

    def __call__(self, image: List[Image.Image], threshold: float) -> List[List[Element]]:
        """Inference using function call interface."""
        return self.infer(image, threshold)


class DeformableDetr(SycamoreObjectDetection):

    @requires_modules("transformers", extra="local-inference")
    def __init__(self, model_name_or_path, device=None, cache: Optional[Cache] = None):
        super().__init__()

        self.labels = [
            "N/A",
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Section-header",
            "Table",
            "Text",
            "Title",
        ]

        self.device = device
        self._model_name_or_path = model_name_or_path
        self.cache = cache

        from sycamore.utils.pytorch_dir import get_pytorch_build_directory

        with fasteners.InterProcessLock(_DETR_LOCK_FILE):
            lockfile = Path(get_pytorch_build_directory("MultiScaleDeformableAttention", False)) / "lock"
            lockfile.unlink(missing_ok=True)

            from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

            LogTime("loading_model", point=True)
            with LogTime("load_model", log_start=True):
                self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
                self.model = DeformableDetrForObjectDetection.from_pretrained(model_name_or_path).to(self._get_device())

    # Note: We wrap this in a function so that we can execute on both the leader and the workers
    # to account for heterogeneous systems. Currently if you pass in an explicit device parameter
    # it will be applied everywhere.
    def _get_device(self) -> str:
        return choose_device(self.device, detr=True)

    def infer(self, images: List[Image.Image], threshold: float, use_cache: bool = False) -> List[List[Element]]:
        if use_cache and self.cache:
            results = self._get_cached_inference(images, threshold)
        else:
            results = self._get_uncached_inference(images, threshold)

        batched_results = []
        for result, image in zip(results, images):
            (w, h) = image.size
            elements = []
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                element = create_element(
                    type=self.labels[label],
                    bbox=BoundingBox(box[0] / w, box[1] / h, box[2] / w, box[3] / h).coordinates,
                    properties={"score": score},
                )
                elements.append(element)
            batched_results.append(elements)
            if self.cache:
                hash_key = self._get_hash_key(image, threshold)
                self.cache.set(hash_key, result)

        return batched_results

    def _get_cached_inference(self, images: List[Image.Image], threshold: float) -> list:
        results = []
        uncached_images = []
        uncached_indices = []

        # First, check the cache for each image
        for index, image in enumerate(images):
            key = self._get_hash_key(image, threshold)
            assert self.cache is not None
            cached_layout = self.cache.get(key)
            if cached_layout:
                logger.info(f"Cache Hit for ImageToJson. Cache hit-rate is {self.cache.get_hit_rate()}")
                results.append(cached_layout)
            else:
                uncached_images.append(image)
                uncached_indices.append(index)
                results.append(None)  # Placeholder for uncached image

        # Process the uncached images in a batch
        if uncached_images:
            processed_images = self._get_uncached_inference(uncached_images, threshold)
            # Store processed images in the cache and update the result list
            for index, processed_img in zip(uncached_indices, processed_images):
                results[index] = processed_img
        return results

    def _get_uncached_inference(self, images: List[Image.Image], threshold: float) -> list:
        import torch

        results = []
        inputs = self.processor(images=images, return_tensors="pt").to(self._get_device())
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1] for image in images])
        results.extend(
            self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)
        )
        for result in results:
            result["scores"] = result["scores"].tolist()
            result["labels"] = result["labels"].tolist()
            result["boxes"] = result["boxes"].tolist()
        return results

    def _get_hash_key(self, image: Image.Image, threshold: float) -> str:
        hash_ctx = Cache.get_hash_context(image.tobytes())
        hash_ctx.update(f"{threshold:.6f}".encode())
        hash_ctx.update(_VERSION.encode())
        return hash_ctx.hexdigest()


@timetrace("OCR")
def extract_ocr(
    images: list[Image.Image],
    elements: list[list[Element]],
    ocr_images: bool = False,
    ocr_model_name: str = "easyocr",
) -> list[list[Element]]:
    if ocr_model_name not in EXTRACTOR_DICT.keys():
        raise ValueError(f"Unknown ocr_model: {ocr_model_name}")
    ocr_model: OCRModel = EXTRACTOR_DICT[ocr_model_name]()
    for i, image in enumerate(images):
        page_elements = elements[i]
        width, height = image.size
        for elem in page_elements:
            if elem.bbox is None:
                continue
            if elem.type == "Picture" and not ocr_images:
                continue
            cropped_image = crop_to_bbox(image, elem.bbox)
            if elem.type == "table":
                tokens = []
                assert isinstance(elem, TableElement)
                for token in ocr_model.get_boxes_and_text(cropped_image):
                    # Shift the BoundingBox to be relative to the whole image.
                    # TODO: We can likely reduce the number of bounding box translations/conversion in the pipeline,
                    #  but for the moment I'm prioritizing clarity over (theoretical) performance, and we have the
                    #  desired invariant that whenever we store bounding boxes they are relative to the entire doc.
                    token["bbox"].translate_self(elem.bbox.x1 * width, elem.bbox.y1 * height).to_relative_self(
                        width, height
                    )
                    tokens.append(token)
                elem.tokens = tokens
            else:
                elem.text_representation = ocr_model.get_text(cropped_image)

    return elements
