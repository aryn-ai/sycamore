from abc import abstractmethod
from io import BytesIO
from PIL import Image
from typing import Any, Union, cast
from sycamore.docset import Document
from sycamore.data import BoundingBox
from sycamore.evaluation.ocr.data import OCREvalDocument


class OCRModel:

    @abstractmethod
    def get_text(self, image: Image.Image) -> str:
        pass

    @abstractmethod
    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        pass


class ExtractOCRFromImage:

    def __init__(self, model: "OCRModel"):
        # model_dict = {"paddle": PaddleOCR(), "easy": EasyOCR(), "tesseract": Tesseract(), "legacy": LegacyOCR()}
        self._model = model

    def apply_model(self, docs: list[Document]) -> list[Document]:
        ans_docs = []
        for doc in docs:
            doc = OCREvalDocument(doc.data)
            assert isinstance(doc, OCREvalDocument), f"Wrong kind of doc: {type(doc)}, {doc}"
            image = doc.image
            doc.pred_text = self._model.get_text(image)
            ans_docs.append(cast(Document, doc))
        return ans_docs

    def __call__(self, docs: list[Document]) -> list[Document]:
        return self.apply_model(docs)


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
        return " ".join(out_list)

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
        return self.pytesseract.image_to_string(image)

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
    def __init__(self):
        from paddleocr import PaddleOCR

        self.reader = PaddleOCR(lang="en", use_gpu=False)

    def get_text(
        self,
        image: Image.Image,
    ) -> str:
        bytearray = BytesIO()
        image.save(bytearray, format="PNG")
        return self.reader.ocr(bytearray.getvalue(), rec=False)

    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        bytearray = BytesIO()
        image.save(bytearray, format="PNG")
        return self.reader.ocr(bytearray.getvalue(), det=False)
