from abc import ABC, abstractmethod
import tempfile
from typing import BinaryIO, Any, List, Tuple

from sycamore.data import Element, BoundingBox
from PIL import Image
import pdf2image

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import open_filename
from typing import cast


class SycamorePDFPartitioner:
    def __init__(self, model_name_or_path):
        self.pdfminer = PDFMinerExtractor()
        self.model = DeformableDetr(model_name_or_path)

    @staticmethod
    def _supplement_text(inferred: List[Element], text: List[Element], threshold: float = 0.5) -> List[Element]:
        # The hungarian finds optimal mapping in bipartite graph in time complexity of n^3, it's too expensive,
        # I feel a naive algorithm is good enough here, for each inferred element from model, we iterate through the
        # text entity extracted by pdfminer, as long as one text entity has bbox IOU greater than threshold, we believe
        # it's a solid mapping.
        for ele in inferred:
            matched = None
            for extracted_region in text:
                if ele.bbox and extracted_region.bbox and ele.bbox.iou(extracted_region.bbox) > threshold:
                    ele.text_representation = extracted_region.text_representation
                    matched = extracted_region
            if matched:
                text.remove(matched)
        return inferred

    def partition_pdf(self, file: BinaryIO, threshold: float = 0.4) -> List[List["Element"]]:
        with tempfile.TemporaryDirectory() as tmp_dir, tempfile.NamedTemporaryFile() as tmp_file:
            filename = tmp_file.name
            tmp_file.write(file.read())
            tmp_file.flush()

            images = pdf2image.convert_from_path(
                filename,
                output_folder=tmp_dir,
                paths_only=True,
            )
            image_paths = cast(List[str], images)
            deformable_layout = [self.model.infer(Image.open(path).convert("RGB"), threshold) for path in image_paths]
            pdfminer_layout = self.pdfminer.extract(filename)
            # page count should be the same
            assert len(pdfminer_layout) == len(deformable_layout)

            for d, p in zip(deformable_layout, pdfminer_layout):
                self._supplement_text(d, p)

            return deformable_layout


class SycamoreObjectDetection(ABC):
    """Wrapper class for the various object detection models."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def infer(self, image: Image, threshold: float) -> List[List[Element]]:
        """Do inference using the wrapped model."""
        pass

    def __call__(self, image: Image, threshold: float) -> List[List[Element]]:
        """Inference using function call interface."""
        return self.infer(image, threshold)


class DeformableDetr(SycamoreObjectDetection):
    def __init__(self, model_name_or_path):
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

        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        # cuda is a hard requirement for deformable detr
        self.model = DeformableDetrForObjectDetection.from_pretrained(model_name_or_path).to("cuda")

    def infer(self, image: Image, threshold: float) -> Any:
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
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
            element = Element()
            element.type = self.labels[label]
            element.bbox = BoundingBox(box[0] / w, box[1] / h, box[2] / w, box[3] / h)
            element.properties = {"score": score}
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
