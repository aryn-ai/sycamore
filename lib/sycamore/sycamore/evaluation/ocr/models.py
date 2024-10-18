from abc import abstractmethod
from io import BytesIO
from PIL import Image
from typing import Any, Union, cast
from sycamore.docset import Document
from sycamore.data import BoundingBox
from sycamore.evaluation.ocr.data import OCREvalDocument
import asyncio


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
    def __init__(self):
        pass

    def get_text(
        self,
        image: Image.Image,
    ) -> str:
        from paddleocr import PaddleOCR as OriginalPaddleOCR

        self.reader = OriginalPaddleOCR(lang="en", use_gpu=False)
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


class Textract(OCRModel):
    def __init__(self):
        pass

    def get_text(
        self,
        image: Image.Image,
    ) -> str:
        from textractor import Textractor

        self.reader = Textractor(profile_name="user")
        return self.reader.detect_document_text(image).text

    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        return []


class LLMOCR(OCRModel):
    def __init__(self):
        from sycamore.evaluation.ocr.llm_ocr import LLMOCR

        self.reader = LLMOCR()

    def get_text(
        self,
        image: Image.Image,
    ) -> str:
        return asyncio.run(self.reader.read_text(image))

    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        return []


class DocTR(OCRModel):
    def __init__(self):
        from doctr.models import ocr_predictor

        self.reader = ocr_predictor(pretrained=True, det_arch="db_resnet50", reco_arch="crnn_mobilenet_v3_large")

    def get_text(
        self,
        image: Image.Image,
    ) -> str:
        from doctr.io import DocumentFile
        import io

        ans_str = ""
        # Create a bytes buffer
        buffer = io.BytesIO()

        # Save the image as PNG to the buffer
        image.save(buffer, format="PNG")

        # Get the byte data
        byte_data = buffer.getvalue()
        doc = DocumentFile.from_images(byte_data)
        result = self.reader(doc).export()
        for page in result["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    for word in line["words"]:
                        ans_str += word["value"] + " "
        return ans_str

    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        return []


class RapidOCR(OCRModel):
    def __init__(self):
        pass

    def get_text(
        self,
        image: Image.Image,
    ) -> str:
        from rapidocr_onnxruntime import RapidOCR as OriginalRapidOCR

        self.reader = OriginalRapidOCR()
        bytearray = BytesIO()
        image.save(bytearray, format="PNG")
        result = self.reader(bytearray.getvalue())
        return ans if result and result[0] and (ans := " ".join(value[1] for value in result[0])) else ""

    def get_boxes(self, image: Image.Image) -> list[Union[dict[str, Any], list]]:
        # from rapidocr_onnxruntime import RapidOCR as OriginalRapidOCR

        # self.reader = OriginalRapidOCR()
        # bytearray = BytesIO()
        # image.save(bytearray, format="PNG")
        # result = self.reader(bytearray.getvalue())
        # return [value[0] for value in result[0]] if result and result[0] else ""
        return []
