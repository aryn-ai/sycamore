import gc
import logging
import os
import tempfile
import tracemalloc
import inspect
from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Callable, Literal, Union, Optional
from itertools import repeat

from tenacity import retry, retry_if_exception, wait_exponential, stop_after_delay
import base64
from PIL import Image
from pypdf import PdfReader

from aryn_sdk.partition import partition_file
from sycamore.data import Element, BoundingBox, TableElement
from sycamore.data.document import DocumentPropertyTypes
from sycamore.data.element import create_element
from sycamore.transforms.table_structure.extract import DEFAULT_TABLE_STRUCTURE_EXTRACTOR
from sycamore.utils import choose_device
from sycamore.utils.element_sort import sort_page
from sycamore.utils.cache import Cache
from sycamore.utils.image_utils import crop_to_bbox, extract_images_from_elements
from sycamore.utils.import_utils import requires_modules
from sycamore.utils.markdown import elements_to_markdown
from sycamore.utils.memory_debugging import display_top, gc_tensor_dump
from sycamore.utils.pdf import convert_from_path_streamed_batched
from sycamore.utils.time_trace import LogTime, timetrace
from sycamore.transforms.text_extraction import TextExtractor, OcrModel, get_text_extractor
from sycamore.transforms.text_extraction.pdf_miner import PdfMinerExtractor

from sycamore.transforms.detr_partitioner_config import (
    ARYN_DETR_MODEL,
    DEFAULT_ARYN_PARTITIONER_ADDRESS,
    DEFAULT_LOCAL_THRESHOLD,
)

logger = logging.getLogger(__name__)
_VERSION = "0.2024.07.24"

_TEN_MINUTES = 600


class ArynPDFPartitionerException(Exception):
    def __init__(self, message, can_retry=False):
        super().__init__(message)
        self.can_retry = can_retry


def _can_retry(e: BaseException) -> bool:
    def make_mypy_happy(e: BaseException):
        import traceback

        # type(e), value=e needed for compatibility before 3.10; after that, just e should work
        logger.warning(f"Automatically retrying because of error: {traceback.format_exception_only(type(e), value=e)}")

    if isinstance(e, ArynPDFPartitionerException):
        # make mypy happy, unneeded with mypy 1.15 + python 3.12
        ex: Optional[BaseException] = None
        assert isinstance(e, BaseException)
        ex = e
        make_mypy_happy(ex)
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


def elem_to_tok(elem: Element) -> dict[str, Any]:
    d = {"text": elem.text_representation, "bbox": elem.bbox}
    if (vec := elem.data.get("_vector")) is not None:
        d["vector"] = vec
    return d


def _supplement_text(inferred: list[Element], text: list[Element], threshold: float = 0.5) -> list[Element]:
    """
    Associates extracted text with inferred objects. Meant to be called pagewise. Uses complete containment (the
    text's bbox is fully within the inferred object's bbox), IOU (intersection over union), and IOB (intersection
    over bounding box) to determine if a text object is associated with an inferred object. We allow multiple
    detected objects to contain the same text, we are holding on solving this.

    Once all text that can be associated has been, the text representation of the inferred object is updated to
    incorporate its associated text.

    In order to handle list items properly, we treat them as a special case.
    """
    logger.info("running _supplement_text")

    unmatched = text.copy()
    for index_i, i in enumerate(inferred):
        matched = []
        for t in text:
            if (
                i.bbox
                and t.bbox
                and (i.bbox.iou(t.bbox) > threshold or t.bbox.iob(i.bbox) > threshold or i.bbox.contains(t.bbox))
            ):
                matched.append(t)
                if t in unmatched:
                    unmatched.remove(t)
        if matched:
            matches = []
            full_text = []
            font_sizes = []
            is_list_item = i.type == "List-item"
            num_matched = len(matched)
            for m_index, m in enumerate(matched):
                matches.append(m)
                if text_to_add := m.text_representation:
                    if (
                        is_list_item and m_index + 1 < num_matched and text_to_add[-1] == "\n"
                    ):  # special case for list items
                        text_to_add = text_to_add[:-1]
                    full_text.append(text_to_add)
                    if font_size := m.properties.get("font_size"):
                        font_sizes.append(font_size)
            if isinstance(i, TableElement):
                i.tokens = [elem_to_tok(elem) for elem in matches]

            i.data["text_representation"] = " ".join(full_text)
            i.properties["font_size"] = sum(font_sizes) / len(font_sizes) if font_sizes else None
    return inferred + unmatched


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

    def partition_pdf(
        self,
        file: BinaryIO,
        threshold: Union[float, Literal["auto"]] = DEFAULT_LOCAL_THRESHOLD,
        use_ocr=False,
        ocr_model: Union[str, OcrModel] = "easyocr",
        per_element_ocr=True,
        extract_table_structure=False,
        table_structure_extractor=None,
        table_extraction_options: dict = {},
        extract_images=False,
        extract_image_format: str = "PPM",
        batch_size: int = 1,
        use_partitioning_service=True,
        aryn_api_key: str = "",
        aryn_partitioner_address=DEFAULT_ARYN_PARTITIONER_ADDRESS,
        use_cache=False,
        pages_per_call: int = -1,
        output_format: Optional[str] = None,
        text_extraction_options: dict[str, Any] = {},
        source: str = "",
        output_label_options: dict[str, Any] = {},
        sort_mode: Optional[str] = None,
        **kwargs,
    ) -> list[Element]:
        if use_partitioning_service:
            assert aryn_api_key != ""

            return self._partition_remote(
                file=file,
                aryn_api_key=aryn_api_key,
                aryn_partitioner_address=aryn_partitioner_address,
                threshold=threshold,
                use_ocr=use_ocr,
                extract_table_structure=extract_table_structure,
                extract_images=extract_images,
                extract_image_format=extract_image_format,
                pages_per_call=pages_per_call,
                output_format=output_format,
                source=source,
                output_label_options=output_label_options,
                **kwargs,
            )
        else:
            if isinstance(threshold, str):
                raise ValueError("Auto threshold is only supported with Aryn DocParse.")

            temp = self._partition_pdf_batched(
                file=file,
                threshold=threshold,
                use_ocr=use_ocr,
                ocr_model=ocr_model,
                per_element_ocr=per_element_ocr,
                extract_table_structure=extract_table_structure,
                table_structure_extractor=table_structure_extractor,
                table_extraction_options=table_extraction_options,
                extract_images=extract_images,
                extract_image_format=extract_image_format,
                batch_size=batch_size,
                use_cache=use_cache,
                text_extraction_options=text_extraction_options,
            )
            elements = []
            for i, r in enumerate(temp):
                page = []
                for ele in r:
                    ele.properties[DocumentPropertyTypes.PAGE_NUMBER] = i + 1
                    page.append(ele)
                if output_label_options.get("promote_title", False):
                    from sycamore.utils.pdf_utils import promote_title

                    if title_candidate_elements := output_label_options.get("title_candidate_elements"):
                        promote_title(page, title_candidate_elements)
                    else:
                        promote_title(page)
                sort_page(page, mode=sort_mode)
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
    def _call_remote_partitioner(file: BinaryIO, **kwargs) -> list[Element]:
        # Get accepted parameters from partition_file function
        partition_params = set(inspect.signature(partition_file).parameters.keys())

        source = f"sycamore-{source_kwarg}" if (source_kwarg := kwargs.pop("source", "")) else "sycamore"
        extra_headers = kwargs.pop("extra_headers", {})
        extra_headers["X-Aryn-Origin"] = source

        # Filter kwargs to only include parameters accepted by partition_file
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in partition_params}

        try:
            file.seek(0)
            response_json = partition_file(file, extra_headers=extra_headers, **filtered_kwargs)
        except Exception as e:
            raise ArynPDFPartitionerException(f"Error calling Aryn DocParse: {e}", can_retry=True)
        if (kwargs.get("output_format") == "markdown") and ((md := response_json.get("markdown")) is not None):
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
        extract_table_structure: bool = False,
        extract_images: bool = False,
        extract_image_format: str = "PPM",
        pages_per_call: int = -1,
        output_format: Optional[str] = None,
        source: str = "",
        output_label_options: dict[str, Any] = {},
        **kwargs,
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
                    docparse_url=aryn_partitioner_address,
                    threshold=threshold,
                    use_ocr=use_ocr,
                    extract_table_structure=extract_table_structure,
                    extract_images=extract_images,
                    extract_image_format=extract_image_format,
                    selected_pages=[[low, min(high, page_count)]],
                    output_format=output_format,
                    source=source,
                    output_label_options=output_label_options,
                    **kwargs,
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
        ocr_model: Union[str, OcrModel] = "easyocr",
        per_element_ocr: bool = True,
        extract_table_structure: bool = False,
        table_structure_extractor=None,
        table_extraction_options: dict = {},
        extract_images: bool = False,
        extract_image_format: str = "PPM",
        batch_size: int = 1,
        use_cache=False,
        text_extraction_options: dict[str, Any] = {},
    ) -> list[list["Element"]]:
        self._init_model()

        LogTime("partition_start", point=True)
        # We use NamedTemporaryFile just for the file name.  On Windows,
        # if we use the opened file, we can't open it a second time.
        pdffile = tempfile.NamedTemporaryFile(prefix="detr-pdf-input-", delete=False)
        try:
            pdffile.file.close()
            with LogTime("write_pdf"):
                file_hash = Cache.copy_and_hash_file(file, pdffile.name)
                logger.info(f"Wrote {pdffile.name}")
            return self._partition_pdf_batched_named(
                pdffile.name,
                file_hash.hexdigest(),
                threshold,
                use_ocr,
                ocr_model,
                per_element_ocr,
                extract_table_structure,
                table_structure_extractor,
                table_extraction_options,
                extract_images,
                extract_image_format,
                batch_size,
                use_cache,
                text_extraction_options,
            )
        finally:
            os.unlink(pdffile.name)

    def _partition_pdf_batched_named(
        self,
        filename: str,
        hash_key: str,
        threshold: float = DEFAULT_LOCAL_THRESHOLD,
        use_ocr: bool = False,
        ocr_model: Union[str, OcrModel] = "easyocr",
        per_element_ocr: bool = True,
        extract_table_structure=False,
        table_structure_extractor=None,
        table_extraction_options: dict = {},
        extract_images=False,
        extract_image_format="PPM",
        batch_size: int = 1,
        use_cache=False,
        text_extraction_options: dict[str, Any] = {},
    ) -> list[list["Element"]]:
        self._init_model()

        if extract_table_structure and not table_structure_extractor:
            table_structure_extractor = DEFAULT_TABLE_STRUCTURE_EXTRACTOR(device=self.device)

        text_extractor: TextExtractor

        if use_ocr:
            if isinstance(ocr_model, OcrModel):
                text_extractor = ocr_model
            else:
                text_extractor = get_text_extractor(ocr_model, **text_extraction_options)
            text_generator: Any = repeat(None)
        else:
            text_extractor = get_text_extractor("pdfminer", **text_extraction_options)
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
                ocr_model=ocr_model,
                per_element_ocr=per_element_ocr,
                extract_table_structure=extract_table_structure,
                table_structure_extractor=table_structure_extractor,
                table_extraction_options=table_extraction_options,
                extract_images=extract_images,
                extract_image_format=extract_image_format,
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

    @staticmethod
    def _run_text_extractor_document(
        file_name: str,
        hash_key: str,
        use_cache: bool,
        use_ocr: bool,
        text_extractor_model: Union[str, OcrModel],
        text_extraction_options: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
    ):
        kwargs = {"images": images}
        if isinstance(text_extractor_model, OcrModel):
            model: TextExtractor = text_extractor_model
        else:
            model = get_text_extractor("pdfminer" if not use_ocr else text_extractor_model, **text_extraction_options)
        with LogTime("text_extract", log_start=True):
            extracted_layout = model.extract_document(file_name, hash_key, use_cache, **kwargs)
        return extracted_layout

    def process_batch_inference(
        self,
        batch: list[Image.Image],
        *,
        threshold: float,
        use_cache: bool,
        use_ocr: bool,
        ocr_model: Union[str, OcrModel],
        per_element_ocr: bool,
        extractor_inputs: Optional[Any] = None,
        text_extractor: Optional[TextExtractor] = None,
        supplement_text_fn: Callable[[list[Element], list[Element]], list[Element]] = _supplement_text,
    ) -> Any:
        self._init_model()
        with LogTime("infer"):
            assert self.model is not None
            deformable_layout = self.model.infer(batch, threshold, use_cache)

        if not extractor_inputs:
            extractor_inputs = batch
        gc_tensor_dump()
        assert len(deformable_layout) == len(batch)
        if use_ocr and per_element_ocr:
            extract_ocr(
                batch,
                deformable_layout,
                ocr_model=ocr_model,
            )
        elif text_extractor is not None:
            extracted_pages = []
            with LogTime("text_extraction"):
                for i, page_data in enumerate(extractor_inputs):
                    if isinstance(page_data, dict):
                        width, height = page_data.get("dimensions")
                        page = text_extractor.parse_output(page_data.get("data"), width, height)
                    else:
                        page = text_extractor.extract_page(page_data)
                    extracted_pages.append(page)
            assert len(extracted_pages) == len(deformable_layout)
            with LogTime("text_supplement"):
                for d, p in zip(deformable_layout, extracted_pages):
                    supplement_text_fn(d, p)
        return deformable_layout

    def process_batch_extraction(
        self,
        batch: list[Image.Image],
        *,
        deformable_layout: list[list[Element]],
        extract_table_structure: bool,
        table_structure_extractor,
        table_extraction_options: dict,
        extract_images: bool,
        extract_image_format: str = "PPM",
    ) -> Any:
        if extract_table_structure:
            with LogTime("extract_table_structure_batch"):
                if table_structure_extractor is None:
                    table_structure_extractor = DEFAULT_TABLE_STRUCTURE_EXTRACTOR(device=self.device)
                for i, page_elements in enumerate(deformable_layout):
                    image = batch[i]
                    for j in range(len(page_elements)):
                        element = page_elements[j]
                        if isinstance(element, TableElement):
                            page_elements[j] = table_structure_extractor.extract(
                                element, image, **table_extraction_options
                            )

        if extract_images:
            with LogTime("extract_images_batch"):
                for i, page_elements in enumerate(deformable_layout):
                    image = batch[i]
                    deformable_layout[i] = extract_images_from_elements(page_elements, image, extract_image_format)
        return deformable_layout

    def process_batch(
        self,
        batch: list[Image.Image],
        *,
        threshold: float,
        text_extractor: TextExtractor,
        extractor_inputs: Any,
        use_ocr: bool,
        ocr_model: Union[str, OcrModel],
        per_element_ocr: bool,
        extract_table_structure: bool,
        table_structure_extractor,
        table_extraction_options: dict,
        extract_images: bool,
        extract_image_format: str,
        use_cache,
        skip_empty_tables: bool = False,
        supplement_text_fn: Callable[[list[Element], list[Element]], list[Element]] = _supplement_text,
    ) -> list[list[Element]]:
        deformable_layout = self.process_batch_inference(
            batch,
            threshold=threshold,
            use_cache=use_cache,
            use_ocr=use_ocr,
            ocr_model=ocr_model,
            per_element_ocr=per_element_ocr,
            text_extractor=text_extractor,
            supplement_text_fn=supplement_text_fn,
            extractor_inputs=extractor_inputs,
        )
        if extract_table_structure or extract_images:
            return self.process_batch_extraction(
                batch,
                deformable_layout=deformable_layout,
                extract_table_structure=extract_table_structure,
                table_structure_extractor=table_structure_extractor,
                table_extraction_options=table_extraction_options,
                extract_images=extract_images,
                extract_image_format=extract_image_format,
            )
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

        from transformers import AutoImageProcessor
        from sycamore.utils.model_load import load_deformable_detr

        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = load_deformable_detr(model_name_or_path, self._get_device())

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
        with torch.no_grad():
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
    ocr_model: Union[str, OcrModel] = "easyocr",
    text_extraction_options: dict[str, Any] = {},
) -> list[list[Element]]:
    ocr_model_obj: OcrModel
    if isinstance(ocr_model, OcrModel):
        ocr_model_obj = ocr_model
    else:
        extractor = get_text_extractor(ocr_model, **text_extraction_options)
        if not isinstance(extractor, OcrModel):
            raise TypeError(f"Unexpected OCR model type {ocr_model}")
        ocr_model_obj = extractor

    for i, image in enumerate(images):
        page_elements = elements[i]
        width, height = image.size
        for elem in page_elements:
            if elem.bbox is None:
                continue
            cropped_image = crop_to_bbox(image, elem.bbox, padding=0)
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
                elem.text_representation, elem.properties["font_size"] = ocr_model_obj.get_text(cropped_image)

    return elements
