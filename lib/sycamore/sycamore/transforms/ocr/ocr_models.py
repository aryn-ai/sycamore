from abc import abstractmethod
from io import BytesIO
from PIL import Image
from typing import Any, Union
from sycamore.data import BoundingBox


class OCRModel:

    @abstractmethod
    def get_text(self, image: Image.Image) -> str:
        pass

    @abstractmethod
    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        pass


class EasyOCR(OCRModel):
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

    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        raw_results = self.reader.readtext(image_bytes.getvalue())

        out: list[Union[dict[str, Any], list]] = []
        for res in raw_results:
            raw_bbox = res[0]
            text = res[1]
            out.append(
                {"bbox": BoundingBox(raw_bbox[0][0], raw_bbox[0][1], raw_bbox[2][0], raw_bbox[2][1]), "text": text}
            )

        return out


class Tesseract(OCRModel):
    def __init__(self):
        import pytesseract

        self.pytesseract = pytesseract

    def get_text(self, image: Image.Image) -> str:
        val = self.pytesseract.image_to_string(image)
        return val

    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        return [self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)]


class LegacyOCR(OCRModel):
    """Match existing behavior where we use tesseract for the main text and EasyOCR for tables."""

    def __init__(self):
        self.tesseract = Tesseract()
        self.easy_ocr = EasyOCR()

    def get_text(self, image: Image.Image) -> str:
        return self.tesseract.get_text(image)

    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        return self.easy_ocr.get_boxes(image)


class PaddleOCR(OCRModel):
    def __init__(self, use_gpu=True, language="en"):
        self.use_gpu = use_gpu
        self.language = language

    def get_text(
        self,
        image: Image.Image,
    ) -> str:
        from paddleocr import PaddleOCR

        self.reader = PaddleOCR(lang=self.language, use_gpu=self.use_gpu)
        bytearray = BytesIO()
        image.save(bytearray, format="PNG")
        result = self.reader.ocr(bytearray.getvalue(), rec=True, det=True, cls=False)
        return ans if result and result[0] and (ans := " ".join(value[1][0] for value in result[0])) else ""

    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        from paddleocr import PaddleOCR

        self.reader = PaddleOCR(lang="en", use_gpu=False)
        bytearray = BytesIO()
        image.save(bytearray, format="PNG")
        result = self.reader.ocr(bytearray.getvalue(), rec=False, det=True, cls=False)
        return result[0]
