import gc
import logging
import os
import tempfile
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO, IOBase
from typing import cast, Any, BinaryIO, List, Tuple, Union

import requests
import json
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_delay
import base64
import pdf2image
import pytesseract
import torch
from PIL import Image
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import open_filename

from sycamore.data import Element, BoundingBox, ImageElement, TableElement
from sycamore.data.element import create_element
from sycamore.transforms.table_structure.extract import DEFAULT_TABLE_STRUCTURE_EXTRACTOR
from sycamore.utils import choose_device
from sycamore.utils.cache import Cache, DiskCache
from sycamore.utils.image_utils import crop_to_bbox, image_to_bytes
from sycamore.utils.memory_debugging import display_top, gc_tensor_dump
from sycamore.utils.pdf import convert_from_path_streamed_batched
from sycamore.utils.time_trace import LogTime, timetrace


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


pdf_miner_cache = DiskCache(os.path.join(tempfile.gettempdir(), "SycamoreCache/PDFMinerCache"))


class ArynPDFPartitioner:
    """
    This class contains the implementation of PDF partitioning using a Deformable DETR model.

    This is an implementation class. Callers looking to partition a DocSet should use the
    ArynPartitioner class.
    """

    def __init__(self, model_name_or_path=ARYN_DETR_MODEL, device=None):
        """
        Initializes the ArynPDFPartitioner and underlying DETR model.

        Args:
            model_name_or_path: The HuggingFace coordinates or local path to the DeformableDETR weights to use.
            device: The device on which to run the model.
        """
        self.device = device
        self.model = DeformableDetr(model_name_or_path, device)
        self.ocr_table_reader = None

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

                i.text_representation = " ".join(full_text)

        return inferred + unmatched

    def partition_pdf(
        self,
        file: BinaryIO,
        threshold: float = 0.4,
        use_ocr=False,
        ocr_images=False,
        ocr_tables=False,
        extract_table_structure=False,
        table_structure_extractor=None,
        extract_images=False,
        batch_size: int = 1,
        batch_at_a_time=True,
        local=False,
        aryn_api_key: str = "",
        aryn_partitioner_address=DEFAULT_ARYN_PARTITIONER_ADDRESS,
        use_cache=False,
    ) -> List[Element]:
        if not local:
            return self._partition_remote(
                file=file,
                aryn_api_key=aryn_api_key,
                aryn_partitioner_address=aryn_partitioner_address,
                threshold=threshold,
                use_ocr=use_ocr,
                ocr_images=ocr_images,
                ocr_tables=ocr_tables,
                extract_table_structure=extract_table_structure,
                extract_images=extract_images,
            )
        else:
            if batch_at_a_time:
                temp = self._partition_pdf_batched(
                    file=file,
                    threshold=threshold,
                    use_ocr=use_ocr,
                    ocr_images=ocr_images,
                    ocr_tables=ocr_tables,
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
                    ocr_tables=ocr_tables,
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
    def _partition_remote(
        file: BinaryIO,
        aryn_api_key: str,
        aryn_partitioner_address=DEFAULT_ARYN_PARTITIONER_ADDRESS,
        threshold: float = 0.4,
        use_ocr: bool = False,
        ocr_images: bool = False,
        ocr_tables: bool = False,
        extract_table_structure: bool = False,
        extract_images: bool = False,
    ) -> List[Element]:
        options = {
            "threshold": threshold,
            "use_ocr": use_ocr,
            "ocr_images": ocr_images,
            "ocr_tables": ocr_tables,
            "extract_table_structure": extract_table_structure,
            "extract_images": extract_images,
        }

        files: Mapping = {"pdf": file, "options": json.dumps(options).encode("utf-8")}
        header = {"Authorization": f"Bearer {aryn_api_key}"}

        response = requests.post(aryn_partitioner_address, files=files, headers=header)

        if response.status_code != 200:
            if response.status_code == 500:
                raise ArynPDFPartitionerException(
                    f"Error: status_code: {response.status_code}, reason: {response.text}", can_retry=True
                )
            raise ArynPDFPartitionerException(f"Error: status_code: {response.status_code}, reason: {response.text}")

        response_json = response.json()
        if isinstance(response_json, dict):
            response_json = response_json.get("elements")
        elements = []
        for element_json in response_json:
            element = create_element(**element_json)
            if element.binary_representation:
                element.binary_representation = base64.b64decode(element.binary_representation)
            elements.append(element)

        return elements

    def _partition_pdf_sequenced(
        self,
        file: BinaryIO,
        threshold: float = 0.4,
        use_ocr=False,
        ocr_images=False,
        ocr_tables=False,
        extract_table_structure=False,
        table_structure_extractor=None,
        extract_images=False,
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
           ocr_tables: If set with use_ocr, will attempt to OCR regions on the document identified as tables.
           extract_table_structure: If true, runs a separate table extraction model to extract cells from
             regions of the document identified as tables.
           table_structure_extractor: The table extraction implementaion to use when extract_table_structure is True.
           extract_images: If true, crops each region identified as an image and
             attaches it to the associated ImageElement.

        Returns:
           A list of lists of Elements. Each sublist corresponds to a page in the original PDF.
        """
        import easyocr

        if not table_structure_extractor:
            table_structure_extractor = DEFAULT_TABLE_STRUCTURE_EXTRACTOR(device=self.device)

        LogTime("partition_start", point=True)
        with LogTime("convert2bytes"):
            images: list[Image.Image] = pdf2image.convert_from_bytes(file.read())

        with LogTime("toRGB"):
            images = [im.convert("RGB") for im in images]

        batches = _batchify(images, batch_size)
        deformable_layout = []
        with LogTime("all_batches"):
            for i, batch in enumerate(batches):
                with LogTime(f"infer_one_batch {i}/{len(images) / batch_size}"):
                    deformable_layout += self.model.infer(batch, threshold)

        if use_ocr:
            with LogTime("ocr"):
                if self.ocr_table_reader is None:
                    self.ocr_table_reader = easyocr.Reader(["en"])

                extract_ocr(
                    images,
                    deformable_layout,
                    ocr_images=ocr_images,
                    ocr_tables=ocr_tables,
                    table_reader=self.ocr_table_reader,
                )
        else:
            with LogTime("pdfminer"):
                pdfminer = PDFMinerExtractor()
                # The cast here is to make mypy happy. PDFMiner expects IOBase,
                # but typing.BinaryIO doesn't extend from it. BytesIO
                # (the concrete class) implements both.
                file_name = cast(IOBase, file)
                hash_key = Cache.get_hash_key(file_name.read())
                with LogTime("pdfminer_extract", log_start=True):
                    pdfminer_layout = pdfminer.extract(file_name, hash_key, use_cache)
                # page count should be the same
                assert len(pdfminer_layout) == len(deformable_layout)

                with LogTime("pdfminer_supplement"):
                    for d, p in zip(deformable_layout, pdfminer_layout):
                        self._supplement_text(d, p)

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
        use_ocr=False,
        ocr_images=False,
        ocr_tables=False,
        extract_table_structure=False,
        table_structure_extractor=None,
        extract_images=False,
        batch_size: int = 1,
        use_cache=False,
    ) -> List[List["Element"]]:
        LogTime("partition_start", point=True)
        with tempfile.NamedTemporaryFile(prefix="detr-pdf-input-") as pdffile:
            with LogTime("write_pdf"):
                data = file.read()
                hash_key = Cache.get_hash_key(data)
                data_len = len(data)
                pdffile.write(data)
                del data
                pdffile.flush()
                logging.info(f"Wrote {pdffile.name}")
            stat = os.stat(pdffile.name)
            assert stat.st_size == data_len
            return self._partition_pdf_batched_named(
                pdffile.name,
                hash_key,
                threshold,
                use_ocr,
                ocr_images,
                ocr_tables,
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
        use_ocr=False,
        ocr_images=False,
        ocr_tables=False,
        extract_table_structure=False,
        table_structure_extractor=None,
        extract_images=False,
        batch_size: int = 1,
        use_cache=False,
    ) -> List[List["Element"]]:
        if extract_table_structure and not table_structure_extractor:
            table_structure_extractor = DEFAULT_TABLE_STRUCTURE_EXTRACTOR(device=self.device)

        pdfminer = None
        exec = ProcessPoolExecutor(max_workers=1)
        if not use_ocr:
            with LogTime("start_pdfminer", log_start=True):
                pdfminer = exec.submit(self._run_pdfminer, filename, hash_key, use_cache)

        deformable_layout = []
        if tracemalloc.is_tracing():
            before = tracemalloc.take_snapshot()
        for i in convert_from_path_streamed_batched(filename, batch_size):
            parts = self.process_batch(
                i,
                threshold=threshold,
                use_ocr=use_ocr,
                ocr_images=ocr_images,
                ocr_tables=ocr_tables,
                extract_table_structure=extract_table_structure,
                table_structure_extractor=table_structure_extractor,
                extract_images=extract_images,
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

        if pdfminer is not None:
            with LogTime("wait_for_pdfminer", log_start=True):
                pdfminer_layout = pdfminer.result()
            assert len(pdfminer_layout) == len(deformable_layout), f"{len(pdfminer_layout)} vs {len(deformable_layout)}"
            with LogTime("pdfminer_supplement"):
                for d, p in zip(deformable_layout, pdfminer_layout):
                    self._supplement_text(d, p)

        if tracemalloc.is_tracing():
            (current, peak) = tracemalloc.get_traced_memory()
            logging.info(f"Memory Usage current={current} peak={peak}")
            top = tracemalloc.take_snapshot()
            display_top(top)
        return deformable_layout

    @staticmethod
    def _run_pdfminer(pdf_path, hash_key, use_cache):
        pdfminer = PDFMinerExtractor()
        with LogTime("pdfminer_extract", log_start=True):
            pdfminer_layout = pdfminer.extract(pdf_path, hash_key, use_cache)

        return pdfminer_layout

    def process_batch(
        self,
        batch: list[Image.Image],
        threshold,
        use_ocr,
        ocr_images,
        ocr_tables,
        extract_table_structure,
        table_structure_extractor,
        extract_images,
    ) -> Any:
        import easyocr

        with LogTime("infer"):
            deformable_layout = self.model.infer(batch, threshold)

        gc_tensor_dump()
        assert len(deformable_layout) == len(batch)

        if use_ocr:
            with LogTime("ocr"):
                if self.ocr_table_reader is None:
                    self.ocr_table_reader = easyocr.Reader(["en"])

                extract_ocr(
                    batch,
                    deformable_layout,
                    ocr_images=ocr_images,
                    ocr_tables=ocr_tables,
                    table_reader=self.ocr_table_reader,
                )
        # else pdfminer happens in parent since it is whole document.

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
    def __init__(self, model_name_or_path, device=None):
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

    def infer(self, images: List[Image.Image], threshold: float) -> List[List[Element]]:
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
        return batched_results


class PDFMinerExtractor:
    def __init__(self):
        rm = PDFResourceManager()
        param = LAParams()
        self.device = PDFPageAggregator(rm, laparams=param)
        self.interpreter = PDFPageInterpreter(rm, self.device)

    def _open_pdfminer_pages_generator(self, fp: BinaryIO):
        pages = PDFPage.get_pages(fp)
        for page in pages:
            self.interpreter.process_page(page)
            page_layout = self.device.get_result()
            yield page, page_layout

    @staticmethod
    def _convert_bbox_coordinates(
        rect: Tuple[float, float, float, float],
        height: float,
    ) -> Tuple[float, float, float, float]:
        """
        pdf coordinates are different, bottom left is origin, also two diagonal points defining a rectangle is
        (bottom left, upper right), for details, refer
        https://www.leadtools.com/help/leadtools/v19/dh/to/pdf-topics-pdfcoordinatesystem.html
        """
        x1, y2, x2, y1 = rect
        y1 = height - y1
        y2 = height - y2
        return x1, y1, x2, y2

    def extract(self, filename: Union[str, IOBase], hash_key: str, use_cache=False) -> List[List[Element]]:
        # The naming is slightly confusing, but `open_filename` accepts either
        # a filename (str) or a file-like object (IOBase)

        cached_result = pdf_miner_cache.get(hash_key) if use_cache else None
        if cached_result:
            logging.info("Cache Hit for PDFMiner. Getting the result from cache.")
            return cached_result
        else:
            with open_filename(filename, "rb") as fp:
                fp = cast(BinaryIO, fp)
                pages = []
                for page, page_layout in self._open_pdfminer_pages_generator(fp):
                    width = page_layout.width
                    height = page_layout.height
                    texts: List[Element] = []
                    for obj in page_layout:
                        x1, y1, x2, y2 = self._convert_bbox_coordinates(obj.bbox, height)

                        if hasattr(obj, "get_text"):
                            text = Element()
                            text.type = "text"
                            text.bbox = BoundingBox(x1 / width, y1 / height, x2 / width, y2 / height)
                            text.text_representation = obj.get_text()
                            if text.text_representation:
                                texts.append(text)

                    pages.append(texts)
                if use_cache:
                    logging.info("Cache Miss for PDFMiner. Storing the result to the cache.")
                    pdf_miner_cache.set(hash_key, pages)
                return pages


@timetrace("OCR")
def extract_ocr(
    images: list[Image.Image], elements: list[list[Element]], ocr_images=False, ocr_tables=False, table_reader=None
) -> list[list[Element]]:
    for i, image in enumerate(images):
        width, height = image.size

        page_elements = elements[i]

        for elem in page_elements:
            if elem.bbox is None:
                continue
            if elem.type == "Picture" and not ocr_images:
                continue
            # elif elem.type == "table" and not ocr_tables:
            #     continue
            elif elem.type == "table":
                assert isinstance(elem, TableElement)
                extract_table_ocr(image, elem, reader=table_reader)
                continue

            crop_box = (elem.bbox.x1 * width, elem.bbox.y1 * height, elem.bbox.x2 * width, elem.bbox.y2 * height)
            cropped_image = image.crop(crop_box)

            # TODO: Do we want to switch to easyocr here too?
            text = pytesseract.image_to_string(cropped_image)

            elem.text_representation = text

    return elements


def extract_table_ocr(image: Image.Image, elem: TableElement, reader):
    width, height = image.size

    assert elem.bbox is not None
    crop_box = (elem.bbox.x1 * width, elem.bbox.y1 * height, elem.bbox.x2 * width, elem.bbox.y2 * height)
    cropped_image = image.crop(crop_box)
    image_bytes = BytesIO()
    cropped_image.save(image_bytes, format="PNG")

    # TODO: support more languages
    results = reader.readtext(image_bytes.getvalue())

    tokens = []

    for res in results:
        raw_bbox = res[0]
        text = res[1]

        token = {"bbox": BoundingBox(raw_bbox[0][0], raw_bbox[0][1], raw_bbox[2][0], raw_bbox[2][1]), "text": text}

        # Shift the BoundingBox to be relative to the whole image.
        # TODO: We can likely reduce the number of bounding box translations/conversion in the pipeline,
        #  but for the moment I'm prioritizing clarity over (theoretical) performance, and we have the
        #  desired invariant that whenever we store bounding boxes they are relative to the entire doc.
        token["bbox"].translate_self(crop_box[0], crop_box[1]).to_relative_self(width, height)
        tokens.append(token)

    elem.tokens = tokens
