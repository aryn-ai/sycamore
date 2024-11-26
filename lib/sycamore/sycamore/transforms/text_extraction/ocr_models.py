from abc import abstractmethod
from PIL import Image
from typing import Any, Union, TYPE_CHECKING, Optional
from sycamore.data import BoundingBox, Element
from sycamore.utils.cache import DiskCache
from pathlib import Path
from io import IOBase, BytesIO
from sycamore.utils.pdf import pdf_to_image_files
from sycamore.utils.import_utils import requires_modules
from sycamore.transforms.text_extraction.text_extractor import TextExtractor
import logging
from sycamore.utils.time_trace import timetrace
import tempfile

if TYPE_CHECKING:
    from pdfminer.pdfpage import PDFPage

# TODO: Add cache support for OCR per page
ocr_cache = DiskCache(str(Path.home() / ".sycamore/OcrCache"))

logger = logging.getLogger(__name__)


class OcrModel(TextExtractor):

    @abstractmethod
    def get_text(self, image: Image.Image) -> tuple[str, Optional[float]]:
        pass

    @abstractmethod
    def get_boxes_and_text(self, image: Image.Image) -> list[dict[str, Any]]:
        pass

    @timetrace("OCRPageEx")
    def extract_page(self, page: Optional[Union["PDFPage", "Image.Image"]]) -> list[Element]:
        assert isinstance(page, Image.Image)
        ocr_output = self.get_boxes_and_text(page)
        width, height = page.size
        return self.parse_output(ocr_output, width, height)

    @timetrace("OCRDocEx")
    def extract_document(
        self, filename: Union[str, IOBase], hash_key: str, use_cache=False, **kwargs
    ) -> list[list[Element]]:
        if use_cache and (cached_result := ocr_cache.get(hash_key)):
            logger.info(f"Cache Hit for OCR. Cache hit-rate is {ocr_cache.get_hit_rate()}")
            return cached_result
        with tempfile.TemporaryDirectory() as tempdirname:  # type: ignore
            assert isinstance(filename, str)
            if images := kwargs.get("images"):
                generator = (image for image in images)
            else:
                generator = pdf_to_image_files(filename, Path(tempdirname))
            pages = []
            for image in generator:
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                ocr_output = self.get_boxes_and_text(image)
                width, height = image.size
                texts: list[Element] = self.parse_output(ocr_output, width, height)
                pages.append(texts)
            if use_cache:
                logger.info("Cache Miss for OCR. Storing the result to the cache.")
                ocr_cache.set(hash_key, pages)
            return pages


class EasyOcr(OcrModel):
    @requires_modules("easyocr", extra="local-inference")
    def __init__(self, lang_list=["en"], **kwargs):
        import easyocr

        self.reader = easyocr.Reader(lang_list=lang_list, **kwargs)

    def get_text(self, image: Image.Image) -> tuple[str, Optional[float]]:
        image_bytes = BytesIO()
        image.save(image_bytes, format="BMP")
        raw_results = self.reader.readtext(image_bytes.getvalue())
        out_list = []
        font_sizes = []
        for res in raw_results:
            text = res[1]
            out_list.append(text)
            font_sizes.append(res[0][2][1] - res[0][0][1])
        val = " ".join(out_list)
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else None
        return val, avg_font_size

    def get_boxes_and_text(self, image: Image.Image) -> list[dict[str, Any]]:
        image_bytes = BytesIO()
        image.save(image_bytes, format="BMP")
        raw_results = self.reader.readtext(image_bytes.getvalue())
        out: list[dict[str, Any]] = []
        for res in raw_results:
            raw_bbox = res[0]
            text = res[1]
            out.append(
                {"bbox": BoundingBox(raw_bbox[0][0], raw_bbox[0][1], raw_bbox[2][0], raw_bbox[2][1]), "text": text}
            )

        return out

    def __name__(self):
        return "EasyOcr"


class Tesseract(OcrModel):
    @requires_modules("pytesseract", extra="local-inference")
    def __init__(self):
        import pytesseract

        self.pytesseract = pytesseract

    def get_text(self, image: Image.Image) -> tuple[str, Optional[float]]:
        val = self.pytesseract.image_to_string(image)
        # font size calculation is not supported for tesseract
        return val, None

    def get_boxes_and_text(self, image: Image.Image) -> list[dict[str, Any]]:
        output_list = []
        base_dict = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
        for value in zip(
            base_dict["left"], base_dict["top"], base_dict["width"], base_dict["height"], base_dict["text"]
        ):
            if value[4]:
                output_list.append(
                    {
                        "bbox": BoundingBox(value[0], value[1], value[0] + value[2], value[1] + value[3]),
                        "text": value[4],
                    }
                )
        return output_list

    def __name__(self):
        return "Tesseract"


class LegacyOcr(OcrModel):
    """Match existing behavior where we use tesseract for the main text and EasyOcr for tables."""

    @requires_modules(["easyocr", "pytesseract"], extra="local-inference")
    def __init__(self):
        self.tesseract = Tesseract()
        self.easy_ocr = EasyOcr()

    def get_text(self, image: Image.Image) -> tuple[str, Optional[float]]:
        # font size calculation is not supported for tesseract
        return self.tesseract.get_text(image)[0], None

    def get_boxes_and_text(self, image: Image.Image) -> list[dict[str, Any]]:
        return self.easy_ocr.get_boxes_and_text(image)

    def __name__(self):
        return "LegacyOcr"


class PaddleOcr(OcrModel):
    # NOTE: Also requires the installation of paddlepaddle or paddlepaddle-gpu
    # depending on your system
    @requires_modules(["paddleocr", "paddle"], extra="local-inference")
    def __init__(self, language="en", slice_kwargs={}):
        from paddleocr import PaddleOCR
        from paddleocr.ppocr.utils.logging import get_logger
        import paddle

        get_logger().setLevel(logging.ERROR)
        self.use_gpu = paddle.device.is_compiled_with_cuda()
        self.language = language
        self.reader = PaddleOCR(lang=self.language, use_gpu=self.use_gpu)
        self.slice_kwargs = slice_kwargs

    def get_text(self, image: Image.Image) -> tuple[str, Optional[float]]:
        bytearray = BytesIO()
        image.save(bytearray, format="BMP")
        result = self.reader.ocr(bytearray.getvalue(), rec=True, det=True, cls=False)
        if result and result[0]:
            text_values = []
            font_sizes = []
            for value in result[0]:
                text_values.append(value[1][0])
                font_sizes.append(value[0][3][1] - value[0][0][1])
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else None
            return " ".join(text_values), avg_font_size
        return "", None

    def set_slicing_parameters(self, image_width, image_height) -> dict[str, Any]:
        slicing_params = {}
        slice_threshold = self.slice_kwargs.get("slice_threshold", 800)  # Only slice big images
        if image_width * image_height > slice_threshold**2:
            merge_x_thres, merge_y_thres = self.slice_kwargs.get("merge_threshold", (10, 10))
            default_horizontal_stride, default_vertical_stride = self.slice_kwargs.get("stride", (300, 500))  # pixels

            # Adjust strides if they exceed image dimensions
            horizontal_stride = min(default_horizontal_stride, image_width)
            vertical_stride = min(default_vertical_stride, image_height)

            # Compile slicing parameters
            slicing_params = {
                "horizontal_stride": horizontal_stride,
                "vertical_stride": vertical_stride,
                "merge_x_thres": merge_x_thres,
                "merge_y_thres": merge_y_thres,
            }

        return slicing_params

    def get_boxes_and_text(self, image: Image.Image) -> list[dict[str, Any]]:
        bytearray = BytesIO()
        image.save(bytearray, format="BMP")
        width, height = image.size
        result = self.reader.ocr(
            bytearray.getvalue(), rec=True, det=True, cls=False, slice=self.set_slicing_parameters(width, height)
        )
        out: list[dict[str, Any]] = []
        if not result or not result[0]:
            return out
        for res in result[0]:
            raw_bbox = res[0]
            text = res[1][0]
            out.append(
                {"bbox": BoundingBox(raw_bbox[0][0], raw_bbox[0][1], raw_bbox[2][0], raw_bbox[2][1]), "text": text}
            )
        return out  # type: ignore

    def __name__(self):
        return "PaddleOcr"
