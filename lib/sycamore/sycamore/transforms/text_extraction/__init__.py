from sycamore.transforms.text_extraction.ocr_models import OCRModel, PaddleOCR, LegacyOCR, Tesseract, EasyOCR
from sycamore.transforms.text_extraction.pdf_miner import PDFMinerExtractor
from sycamore.transforms.text_extraction.text_extractor import TextExtractor
from dataclasses import field
from typing import Dict

__all__ = ["PaddleOCR", "LegacyOCR", "Tesseract", "EasyOCR", "PDFMinerExtractor", "OCRModel", "TextExtractor"]

EXTRACTOR_DICT: Dict = field(
    default_factory=lambda: {
        "paddle": PaddleOCR,
        "legacy": LegacyOCR,
        "tesseract": Tesseract,
        "easyocr": EasyOCR,
        "pdfminer": PDFMinerExtractor,
    }
)
