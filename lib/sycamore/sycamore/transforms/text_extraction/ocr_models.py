from abc import abstractmethod
from io import BytesIO
from PIL import Image
from typing import Any, Union
from sycamore.data import BoundingBox
from typing import Any, Union, List, Dict, cast, BinaryIO
from sycamore.data import BoundingBox, Element
from sycamore.utils.cache import DiskCache
from typing import List, Union
from pathlib import Path
from io import IOBase
from pdfminer.utils import open_filename
import pdf2image
from sycamore.transforms.text_extraction import TextExtractor
import logging

ocr_cache = DiskCache(str(Path.home() / ".sycamore/OCRCache"))

logger = logging.getLogger(__name__)


class OCRModel(TextExtractor):

    @abstractmethod
    def get_text(self, image: Image.Image) -> str:
        pass

    @abstractmethod
    def get_boxes_and_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        pass

    def extract(self, filename: Union[str, IOBase], hash_key: str, use_cache=False) -> List[List[Element]]:
        # The naming is slightly confusing, but `open_filename` accepts either
        # a filename (str) or a file-like object (IOBase)

        cached_result = ocr_cache.get(hash_key) if use_cache else None
        if cached_result:
            logger.info(f"Cache Hit for OCR. Cache hit-rate is {ocr_cache.get_hit_rate()}")
            return cached_result
        else:
            with open_filename(filename, "rb") as fp:
                fp = cast(BinaryIO, fp)
                images = pdf2image.convert_from_bytes(fp.read())
                pages = []
                for image in images:
                    ocr_output = self.get_boxes_and_text(image)
                    texts: List[Element] = []
                    for obj in ocr_output:
                        if obj["bbox"] and not obj["bbox"].is_empty() and obj["text"] and len(obj["text"]) > 0:
                            text = Element()
                            text.type = "text"
                            text.bbox = obj["bbox"]
                            text.text_representation = obj["text"]
                            texts.append(text)

                    pages.append(texts)
                if use_cache:
                    logger.info("Cache Miss for OCR. Storing the result to the cache.")
                    ocr_cache.set(hash_key, pages)
                return pages


# @timetrace("OCR")
# def extract_ocr(
#     images: list[Image.Image],
#     elements: list[list[Element]],
#     ocr_images=False,
#     ocr_tables=False,
#     ocr_model=OCRModel,
# ) -> list[list[Element]]:
#     for i, image in enumerate(images):
#         page_elements = elements[i]
#         width, height = image.size
#         for elem in page_elements:
#             if elem.bbox is None:
#                 continue
#             if elem.type == "Picture" and not ocr_images:
#                 continue
#             cropped_image = crop_to_bbox(image, elem.bbox)
#             if elem.type == "table" and ocr_tables:
#                 tokens = []
#                 assert isinstance(elem, TableElement)
#                 for token in ocr_model.get_boxes_and_text(cropped_image):
#                     # Shift the BoundingBox to be relative to the whole image.
#                     # TODO: We can likely reduce the number of bounding box translations/conversion in the pipeline,
#                     #  but for the moment I'm prioritizing clarity over (theoretical) performance, and we have the
#                     #  desired invariant that whenever we store bounding boxes they are relative to the entire doc.
#                     token["bbox"].translate_self(elem.bbox.x1 * width, elem.bbox.y1 * height).to_relative_self(
#                         width, height
#                     )
#                     tokens.append(token)
#                 elem.tokens = tokens
#             else:
#                 elem.text_representation = ocr_model.get_text(cropped_image)

#     return elements


class EasyOCR(OCRModel):
    @requires_modules("easyocr", extra="local-inference")
    def __init__(self, lang_list=["en"]):
        import easyocr

        self.reader = easyocr.Reader(lang_list=lang_list)

    def get_text(self, image: Image.Image) -> str:
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        raw_results = self.reader.readtext(image_bytes.getvalue())
        out_list = []
        for res in raw_results:
            text = res[1]
            out_list.append(text)
        val = " ".join(out_list)
        return val

    def get_boxes_and_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        raw_results = self.reader.readtext(image_bytes.getvalue())

        out: list[dict[str, Any]] = []
        for res in raw_results:
            raw_bbox = res[0]
            text = res[1]
            out.append(
                {"bbox": BoundingBox(raw_bbox[0][0], raw_bbox[0][1], raw_bbox[2][0], raw_bbox[2][1]), "text": text}
            )

        return out


class Tesseract(OCRModel):
    @requires_modules("pytesseract", extra="local-inference")
    def __init__(self):
        import pytesseract

        self.pytesseract = pytesseract

    def get_text(self, image: Image.Image) -> str:
        val = self.pytesseract.image_to_string(image)
        return val

    def get_boxes_and_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        output_list = []
        base_dict = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
        for value in zip(
            base_dict["left"], base_dict["top"], base_dict["width"], base_dict["height"], base_dict["text"]
        ):
            if value[4] != "":
                output_list.append(
                    {
                        "bbox": BoundingBox(value[0], value[1], value[0] + value[2], value[1] + value[3]),
                        "text": value[4],
                    }
                )
        return output_list


class LegacyOCR(OCRModel):
    """Match existing behavior where we use tesseract for the main text and EasyOCR for tables."""

    @requires_modules(["easyocr", "pytesseract"], extra="local-inference")
    def __init__(self):
        self.tesseract = Tesseract()
        self.easy_ocr = EasyOCR()

    def get_text(self, image: Image.Image) -> str:
        return self.tesseract.get_text(image)

    def get_boxes_and_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        return self.easy_ocr.get_boxes_and_text(image)


class PaddleOCR(OCRModel):
    @requires_modules("paddleocr", extra="local-inference")
    def __init__(self, use_gpu=True, language="en"):
        from paddleocr import PaddleOCR

        self.use_gpu = use_gpu
        self.language = language
        self.reader = PaddleOCR(lang=self.language, use_gpu=self.use_gpu)

    def get_text(
        self,
        image: Image.Image,
    ) -> str:
        bytearray = BytesIO()
        image.save(bytearray, format="PNG")
        result = self.reader.ocr(bytearray.getvalue(), rec=True, det=True, cls=False)
        return ans if result and result[0] and (ans := " ".join(value[1][0] for value in result[0])) else ""

    def get_boxes_and_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        bytearray = BytesIO()
        image.save(bytearray, format="PNG")
        result = self.reader.ocr(bytearray.getvalue(), rec=True, det=True, cls=False)
        out = []
        for res in result[0]:
            raw_bbox = res[0]
            text = res[1][0]
            out.append(
                {"bbox": BoundingBox(raw_bbox[0][0], raw_bbox[0][1], raw_bbox[2][0], raw_bbox[2][1]), "text": text}
            )
        return out  # type: ignore
