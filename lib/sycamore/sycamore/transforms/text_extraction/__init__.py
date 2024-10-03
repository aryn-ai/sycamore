from sycamore.transforms.text_extraction.ocr_models import OcrModel, PaddleOcr, LegacyOcr, Tesseract, EasyOcr
from sycamore.transforms.text_extraction.pdf_miner import PdfMinerExtractor
from sycamore.transforms.text_extraction.text_extractor import TextExtractor

__all__ = ["PaddleOcr", "LegacyOcr", "Tesseract", "EasyOcr", "PdfMinerExtractor", "OcrModel", "TextExtractor"]

EXTRACTOR_DICT = {
    "paddle": PaddleOcr,
    "legacy": LegacyOcr,
    "tesseract": Tesseract,
    "easyocr": EasyOcr,
    "pdfminer": PdfMinerExtractor,
}
