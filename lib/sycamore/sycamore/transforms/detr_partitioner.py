from abc import ABC, abstractmethod
from io import BytesIO
import tempfile
from typing import cast, BinaryIO, List, Tuple

from sycamore.data import Element, BoundingBox, TableElement
from sycamore.data.element import create_element

from sycamore.transforms.table_structure.extract import DEFAULT_TABLE_STRUCTURE_EXTRACTOR
from PIL import Image
import pdf2image

import torch

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import open_filename

import pytesseract

import easyocr


class SycamorePDFPartitioner:
    def __init__(self, model_name_or_path, device=None):
        self.model = DeformableDetr(model_name_or_path, device)

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
        table_structure_extractor=DEFAULT_TABLE_STRUCTURE_EXTRACTOR,
    ) -> List[List["Element"]]:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.NamedTemporaryFile() as tmp_file:
            filename = tmp_file.name
            tmp_file.write(file.read())
            tmp_file.flush()

            image_paths: list[str] = pdf2image.convert_from_path(
                filename,
                output_folder=tmp_dir,
                paths_only=True,
            )
            images = [Image.open(path).convert("RGB") for path in image_paths]

            deformable_layout = [self.model.infer(image, threshold) for image in images]

            if use_ocr:
                extract_ocr(images, deformable_layout, ocr_images=ocr_images, ocr_tables=ocr_tables)
            else:
                pdfminer = PDFMinerExtractor()
                pdfminer_layout = pdfminer.extract(filename)
                # page count should be the same
                assert len(pdfminer_layout) == len(deformable_layout)

                for d, p in zip(deformable_layout, pdfminer_layout):
                    self._supplement_text(d, p)

            if extract_table_structure:
                for i, page_elements in enumerate(deformable_layout):
                    image = images[i]
                    for element in page_elements:
                        if isinstance(element, TableElement):
                            table_structure_extractor.extract(element, image)

            return deformable_layout


class SycamoreObjectDetection(ABC):
    """Wrapper class for the various object detection models."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def infer(self, image: Image, threshold: float) -> List[Element]:
        """Do inference using the wrapped model."""
        pass

    def __call__(self, image: Image, threshold: float) -> List[Element]:
        """Inference using function call interface."""
        return self.infer(image, threshold)


class DeformableDetr(SycamoreObjectDetection):
    def __init__(self, model_name_or_path, device=None):
        super().__init__()
        from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

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

        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = DeformableDetrForObjectDetection.from_pretrained(model_name_or_path).to(self._get_device())

    # Note: We wrap this in a function so that we can execute on both the leader and the workers
    # to account for heterogeneous systems. Currently if you pass in an explicit device parameter
    # it will be applied everywhere.
    def _get_device(self) -> str:
        if self.device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return self.device

    def infer(self, image: Image.Image, threshold: float) -> list[Element]:
        inputs = self.processor(images=image, return_tensors="pt").to(self._get_device())
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[
            0
        ]
        # need to wrap up the results in elements
        elements = []
        (w, h) = image.size
        for score, label, box in zip(
            results["scores"].cpu().detach().numpy(),
            results["labels"].cpu().detach().numpy(),
            results["boxes"].cpu().detach().numpy(),
        ):
            element = create_element(
                type=self.labels[label],
                bbox=BoundingBox(box[0] / w, box[1] / h, box[2] / w, box[3] / h),
                properties={"score": score},
            )
            elements.append(element)
        return elements


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

    def extract(self, filename: str) -> List[List[Element]]:
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
            return pages


def extract_ocr(
    images: list[Image.Image], elements: list[list[Element]], ocr_images=False, ocr_tables=False
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
                extract_table_ocr(image, elem)
                continue

            crop_box = (elem.bbox.x1 * width, elem.bbox.y1 * height, elem.bbox.x2 * width, elem.bbox.y2 * height)
            cropped_image = image.crop(crop_box)

            # TODO: Do we want to switch to easyocr here too?
            text = pytesseract.image_to_string(cropped_image)

            elem.text_representation = text

    return elements


def extract_table_ocr(image: Image.Image, elem: TableElement):
    width, height = image.size

    assert elem.bbox is not None
    crop_box = (elem.bbox.x1 * width, elem.bbox.y1 * height, elem.bbox.x2 * width, elem.bbox.y2 * height)
    cropped_image = image.crop(crop_box)
    image_bytes = BytesIO()
    cropped_image.save(image_bytes, format="PNG")

    # TODO: support more languages
    reader = easyocr.Reader(["en"])
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
