from sycamore.transforms.text_extraction.ocr_models import OcrModel, PaddleOcr, LegacyOcr, Tesseract, EasyOcr
from sycamore.transforms.text_extraction.pdf_miner import PdfMinerExtractor
from sycamore.transforms.text_extraction.text_extractor import TextExtractorBase

EXTRACTOR_DICT = {
    "paddle": PaddleOcr,
    "legacy": LegacyOcr,
    "tesseract": Tesseract,
    "easyocr": EasyOcr,
    "pdfminer": PdfMinerExtractor,
}


def TextExtractor(extractor_type: str, **kwargs):
    if extractor_type not in EXTRACTOR_DICT:
        raise ValueError(f"Invalid TextExtractor type {extractor_type}")

    return EXTRACTOR_DICT[extractor_type](**kwargs)


__all__ = [
    "PaddleOcr",
    "LegacyOcr",
    "Tesseract",
    "EasyOcr",
    "PdfMinerExtractor",
    "OcrModel",
    "TextExtractor",
    "TextExtractorBase",
]
