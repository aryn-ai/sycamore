import gc
import logging
import os
import tempfile
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, BinaryIO, Literal, Union, Optional
from pathlib import Path
import pwd
from itertools import repeat

import requests
import json
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_delay
import base64
from PIL import Image
import fasteners
from pypdf import PdfReader

from sycamore.data import Element, BoundingBox, ImageElement, TableElement
from sycamore.data.document import DocumentPropertyTypes
from sycamore.data.element import create_element
from sycamore.transforms.table_structure.extract import DEFAULT_TABLE_STRUCTURE_EXTRACTOR
from sycamore.utils import choose_device
from sycamore.utils.bbox_sort import bbox_sort_page
from sycamore.utils.cache import Cache
from sycamore.utils.image_utils import crop_to_bbox, image_to_bytes
from sycamore.utils.import_utils import requires_modules
from sycamore.utils.markdown import elements_to_markdown
from sycamore.utils.memory_debugging import display_top, gc_tensor_dump
from sycamore.utils.pdf import convert_from_path_streamed_batched
from sycamore.utils.time_trace import LogTime, timetrace
from sycamore.transforms.text_extraction import TextExtractor, OcrModel, EXTRACTOR_DICT
from sycamore.transforms.text_extraction.pdf_miner import PdfMinerExtractor

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
DEFAULT_LOCAL_THRESHOLD = 0.35


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


def text_elem(text: str) -> Element:
    return Element(
        {
            "type": "Text",
            "properties": {DocumentPropertyTypes.PAGE_NUMBER: 1},
            "text_representation": text,
        }
    )


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
    def _supplement_text(inferred: list[Element], text: list[Element], threshold: float = 0.5) -> list[Element]:
        # We first check IOU between inferred object and pdf miner text object, we also check if a detected object
        # fully contains a pdf miner text object. After that, we combined all texts belonging a detected object and
        # update its text representation. We allow multiple detected objects contain the same text, we hold on solving
        # this.

        unmatched = text.copy()
        for index_i, i in enumerate(inferred):
            matched = []
            for t in text:
                if i.bbox and t.bbox and (i.bbox.iou(t.bbox) > threshold or t.bbox.iob(i.bbox) > threshold or i.bbox.contains(t.bbox)):
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
        threshold: Union[float, Literal["auto"]] = DEFAULT_LOCAL_THRESHOLD,
        use_ocr=False,
        ocr_images=False,
        ocr_model="easyocr",
        per_element_ocr=True,
        extract_table_structure=False,
        table_structure_extractor=None,
        extract_images=False,
        batch_size: int = 1,
        use_partitioning_service=True,
        aryn_api_key: str = "",
        aryn_partitioner_address=DEFAULT_ARYN_PARTITIONER_ADDRESS,
        use_cache=False,
        pages_per_call: int = -1,
        output_format: Optional[str] = None,
    ) -> list[Element]:
        if use_partitioning_service:
            assert aryn_api_key != ""

            return self._partition_remote(
                file=file,
                aryn_api_key=aryn_api_key,
                aryn_partitioner_address=aryn_partitioner_address,
                threshold=threshold,
                use_ocr=use_ocr,
                ocr_images=ocr_images,
                extract_table_structure=extract_table_structure,
                extract_images=extract_images,
                pages_per_call=pages_per_call,
                output_format=output_format,
            )
        else:
            if isinstance(threshold, str):
                raise ValueError("Auto threshold is only supported with the Aryn Partitioning Service.")

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
            elements = []
            for i, r in enumerate(temp):
                page = []
                for ele in r:
                    ele.properties[DocumentPropertyTypes.PAGE_NUMBER] = i + 1
                    page.append(ele)
                bbox_sort_page(page)
                elements.extend(page)
            if output_format == "markdown":
                md = elements_to_markdown(elements)
                return [text_elem(md)]
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
        threshold: Union[float, Literal["auto"]] = "auto",
        use_ocr: bool = False,
        ocr_images: bool = False,
        extract_table_structure: bool = False,
        extract_images: bool = False,
        selected_pages: list = [],
        output_format: Optional[str] = None,
    ) -> list[Element]:
        file.seek(0)
        options = {
            "threshold": threshold,
            "use_ocr": use_ocr,
            "ocr_images": ocr_images,
            "extract_table_structure": extract_table_structure,
            "extract_images": extract_images,
            "selected_pages": selected_pages,
            "source": "sycamore",
        }
        if output_format:
            options["output_format"] = output_format

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
            if (output_format == "markdown") and ((md := response_json.get("markdown")) is not None):
                return [text_elem(md)]
            response_json = response_json.get("elements", [])

        elements = []
        for idx, element_json in enumerate(response_json):
            element = create_element(element_index=idx, **element_json)
            if element.binary_representation:
                element.binary_representation = base64.b64decode(element.binary_representation)
            elements.append(element)

        return elements

    @staticmethod
    def _partition_remote(
        file: BinaryIO,
        aryn_api_key: str,
        aryn_partitioner_address=DEFAULT_ARYN_PARTITIONER_ADDRESS,
        threshold: Union[float, Literal["auto"]] = "auto",
        use_ocr: bool = False,
        ocr_images: bool = False,
        extract_table_structure: bool = False,
        extract_images: bool = False,
        pages_per_call: int = -1,
        output_format: Optional[str] = None,
    ) -> list[Element]:
        page_count = get_page_count(file)

        result: list[Element] = []
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
                    extract_table_structure=extract_table_structure,
                    extract_images=extract_images,
                    selected_pages=[[low, min(high, page_count)]],
                    output_format=output_format,
                )
            )
            low = high + 1
            high += pages_per_call

        return result

    def _partition_pdf_batched(
        self,
        file: BinaryIO,
        threshold: float = DEFAULT_LOCAL_THRESHOLD,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_model: str = "easyocr",
        per_element_ocr: bool = True,
        extract_table_structure: bool = False,
        table_structure_extractor=None,
        extract_images: bool = False,
        batch_size: int = 1,
        use_cache=False,
    ) -> list[list["Element"]]:
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
        threshold: float = DEFAULT_LOCAL_THRESHOLD,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_model: str = "easyocr",
        per_element_ocr: bool = True,
        extract_table_structure=False,
        table_structure_extractor=None,
        extract_images=False,
        batch_size: int = 1,
        use_cache=False,
    ) -> list[list["Element"]]:
        self._init_model()

        if extract_table_structure and not table_structure_extractor:
            table_structure_extractor = DEFAULT_TABLE_STRUCTURE_EXTRACTOR(device=self.device)

        if use_ocr:
            text_extractor = EXTRACTOR_DICT[ocr_model]()
            text_generator: Any = repeat(None)
        else:
            text_extractor = PdfMinerExtractor()
            text_generator = PdfMinerExtractor.pdf_to_pages(filename)
        deformable_layout = []
        if tracemalloc.is_tracing():
            before = tracemalloc.take_snapshot()
        for i in convert_from_path_streamed_batched(filename, batch_size):
            extractor_inputs: Any = None
            try:
                extractor_inputs = [text_generator.__next__() for _ in range(batch_size)]
            except StopIteration:
                raise ValueError("Not enough pages in PDF")
            parts = self.process_batch(
                i,
                threshold=threshold,
                use_ocr=use_ocr,
                text_extractor=text_extractor,
                extractor_inputs=extractor_inputs,
                ocr_images=ocr_images,
                ocr_model=ocr_model,
                per_element_ocr=per_element_ocr,
                extract_table_structure=extract_table_structure,
                table_structure_extractor=table_structure_extractor,
                extract_images=extract_images,
                use_cache=use_cache,
            )
            assert len(parts) == len(i)
            deformable_layout.extend(parts)
        if tracemalloc.is_tracing():
            gc.collect()
            (current, peak) = tracemalloc.get_traced_memory()
            logger.info(f"Memory Usage current={current} peak={peak}")
            after = tracemalloc.take_snapshot()
            top_stats = after.compare_to(before, "lineno")

            print("[ Top 10 differences ]")
            for stat in top_stats[:10]:
                print(stat)
            before = after
            display_top(after)
        return deformable_layout

    def process_batch(
        self,
        batch: list[Image.Image],
        threshold: float,
        text_extractor: TextExtractor,
        extractor_inputs: Any,
        use_ocr,
        ocr_images,
        ocr_model,
        per_element_ocr,
        extract_table_structure,
        table_structure_extractor,
        extract_images,
        use_cache,
    ) -> Any:
        with LogTime("infer"):
            assert self.model is not None
            deformable_layout = self.model.infer(batch, threshold, use_cache)

        if not extractor_inputs:
            extractor_inputs = batch
        gc_tensor_dump()
        assert len(deformable_layout) == len(batch)
        if use_ocr and per_element_ocr:
            with LogTime("Per Element OCR"):
                extract_ocr(
                    batch,
                    deformable_layout,
                    ocr_images=ocr_images,
                    ocr_model=ocr_model,
                )
        else:
            extracted_pages = []
            with LogTime("text_extraction"):
                for i, page_data in enumerate(extractor_inputs):
                    if isinstance(page_data, list):
                        width, height = batch[i].size
                        page = text_extractor.parse_output(page_data, width, height)
                    else:
                        page = text_extractor.extract_page(page_data)
                    extracted_pages.append(page)
            assert len(extracted_pages) == len(deformable_layout)
            with LogTime("text_supplement"):
                for d, p in zip(deformable_layout, extracted_pages):
                    self._supplement_text(d, p)
        if extract_table_structure:
            if table_structure_extractor is None:
                table_structure_extractor = DEFAULT_TABLE_STRUCTURE_EXTRACTOR(device=self.device)
            with LogTime("extract_table_structure_batch"):
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

    @staticmethod
    def _run_text_extractor_document(
        file_name: str,
        hash_key: str,
        use_cache: bool,
        use_ocr: bool,
        ocr_images: bool,
        text_extractor_model: str,
        images: Optional[list[Image.Image]] = None,
    ):
        kwargs = {"ocr_images": ocr_images, "images": images}
        if not use_ocr:
            text_extractor_model = "pdfminer"
        model_cls = EXTRACTOR_DICT.get(text_extractor_model)
        if not model_cls:
            raise ValueError(f"Unknown Text Extractor Model: {text_extractor_model}")
        model = model_cls()
        with LogTime("text_extract", log_start=True):
            extracted_layout = model.extract_document(file_name, hash_key, use_cache, **kwargs)
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
                ocr_model=ocr_model,
            )
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
    def infer(self, image: list[Image.Image], threshold: float) -> list[list[Element]]:
        """Do inference using the wrapped model."""
        pass

    def __call__(self, image: list[Image.Image], threshold: float) -> list[list[Element]]:
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
    # to account for heterogeneous systems. Currently, if you pass in an explicit device parameter
    # it will be applied everywhere.
    def _get_device(self) -> str:
        return choose_device(self.device, detr=True)

    def infer(self, images: list[Image.Image], threshold: float, use_cache: bool = False) -> list[list[Element]]:
        if use_cache and self.cache:
            results = self._get_cached_inference(images, threshold)
        else:
            results = self._get_uncached_inference(images, threshold)

        batched_results = []
        for result, image in zip(results, images):
            (w, h) = image.size
            elements = []
            for idx, (score, label, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
                # Potential fix if negative bbox is causing downstream failures
                # box = [max(0.0, coord) for coord in box]
                element = create_element(
                    element_index=idx,
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

    def _get_cached_inference(self, images: list[Image.Image], threshold: float) -> list:
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

    def _get_uncached_inference(self, images: list[Image.Image], threshold: float) -> list:
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
    ocr_model: str = "easyocr",
) -> list[list[Element]]:
    model_cls = EXTRACTOR_DICT.get(ocr_model)
    if not model_cls:
        raise ValueError(f"Unknown OCR Model: {ocr_model}")
    ocr_model_obj: OcrModel = model_cls()
    for i, image in enumerate(images):
        page_elements = elements[i]
        width, height = image.size
        for elem in page_elements:
            if elem.bbox is None:
                continue
            if elem.type == "Picture" and not ocr_images:
                continue
            cropped_image = crop_to_bbox(image, elem.bbox)
            if 0 in cropped_image.size:
                elem.text_representation = ""
                continue
            if elem.type == "table":
                tokens = []
                assert isinstance(elem, TableElement)
                for token in ocr_model_obj.get_boxes_and_text(cropped_image):
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
                elem.text_representation = ocr_model_obj.get_text(cropped_image)

    return elements
