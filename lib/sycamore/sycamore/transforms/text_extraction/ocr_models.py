from abc import abstractmethod
from PIL import Image
from typing import Any, Union, TYPE_CHECKING, Optional
from sycamore.data import BoundingBox, Element
from pathlib import Path
from io import IOBase, BytesIO
from sycamore.utils.pdf import pdf_to_image_files
from sycamore.utils.import_utils import requires_modules
from sycamore.transforms.text_extraction.text_extractor import TextExtractor
from sycamore.transforms.text_extraction.ocr_cache import (
    get_ocr_cache_manager,
    CacheMissError,
    set_ocr_cache_path,
    OcrCacheManager,
)
import logging
import os
import numpy as np

from sycamore.utils.time_trace import timetrace
import tempfile

if TYPE_CHECKING:
    from pdfminer.pdfpage import PDFPage

logger = logging.getLogger(__name__)


class OcrModel(TextExtractor):
    """Base class for OCR models with caching support."""

    def __init__(self, cache_path: Optional[str] = None, cache_only: bool = False, disable_caching: bool = True):
        """
        Initialize OCR model with caching support.

        Args:
            cache_path: Path to cache directory or S3 URL (e.g., "s3://bucket/path")
                       If None, uses default local cache
            cache_only: If True, only use cached results (raise error on cache miss)
            disable_caching: If True, disable caching completely (default: True)
        """
        self.cache_only = cache_only
        self.disable_caching = disable_caching
        self.cache_manager: Optional[OcrCacheManager] = None
        if not disable_caching:
            if cache_path is not None:
                set_ocr_cache_path(cache_path)
            self.cache_manager = get_ocr_cache_manager()
        else:
            self.cache_manager = None

        self._model_name = str(self.__class__.__name__)
        self._package_names = self._get_package_names()

    @abstractmethod
    def _get_package_names(self) -> list[str]:
        """Return list of package names whose versions should be included in cache key."""
        pass

    @abstractmethod
    def _get_text_impl(self, image: Image.Image, **kwargs) -> tuple[str, Optional[float]]:
        """Implementation of get_text without caching."""
        pass

    @abstractmethod
    def _get_boxes_and_text_impl(self, image: Image.Image, **kwargs) -> list[dict[str, Any]]:
        """Implementation of get_boxes_and_text without caching."""
        pass

    def get_text(self, image: Image.Image, **kwargs) -> tuple[str, Optional[float]]:
        """
        Extract text from image with caching support.

        Args:
            image: PIL Image to process
            **kwargs: Additional arguments passed to the OCR implementation

        Returns:
            Tuple of (text, average_font_size)

        Raises:
            CacheMissError: If cache_only=True and result not in cache
        """
        # If caching is disabled, directly call implementation
        if self.disable_caching or self.cache_manager is None:
            return self._get_text_impl(image, **kwargs)

        try:
            cached_result = self.cache_manager.get(
                image, self._model_name, "get_text", kwargs, self._package_names, self.cache_only
            )
            if cached_result is not None:
                logger.debug(f"Cache hit for {self._model_name}.get_text")
                return cached_result
        except CacheMissError as e:
            if self.cache_only:
                logger.error(f"Cache miss for {self._model_name}.get_text: {e}")
                raise
            else:
                logger.warning(f"Cache miss for {self._model_name}.get_text: {e}")

        # Cache miss or cache disabled
        logger.debug(f"Cache miss for {self._model_name}.get_text, computing result")
        result = self._get_text_impl(image, **kwargs)

        # Cache the result
        self.cache_manager.set(image, self._model_name, "get_text", kwargs, self._package_names, result)

        return result

    def get_boxes_and_text(self, image: Image.Image, **kwargs) -> list[dict[str, Any]]:
        """
        Extract text boxes and text from image with caching support.

        Args:
            image: PIL Image to process
            **kwargs: Additional arguments passed to the OCR implementation

        Returns:
            List of dictionaries with 'bbox' and 'text' keys

        Raises:
            CacheMissError: If cache_only=True and result not in cache
        """
        # If caching is disabled, directly call implementation
        if self.disable_caching or self.cache_manager is None:
            return self._get_boxes_and_text_impl(image, **kwargs)

        try:
            cached_result = self.cache_manager.get(
                image, self._model_name, "get_boxes_and_text", kwargs, self._package_names, self.cache_only
            )
            if cached_result is not None:
                logger.debug(f"Cache hit for {self._model_name}.get_boxes_and_text")
                assert isinstance(cached_result, list), f"Cached result is not a list: {type(cached_result)}"
                cached_result = [
                    {"bbox": BoundingBox(*dict_value["bbox"]), "text": dict_value["text"]}
                    for dict_value in cached_result
                ]
                return cached_result
        except CacheMissError as e:
            if self.cache_only:
                logger.error(f"Cache miss for {self._model_name}.get_boxes_and_text: {e}")
                raise
            else:
                logger.warning(f"Cache miss for {self._model_name}.get_boxes_and_text: {e}")

        # Cache miss or cache disabled
        logger.debug(f"Cache miss for {self._model_name}.get_boxes_and_text, computing result")
        result = self._get_boxes_and_text_impl(image, **kwargs)
        jsonable_result = [
            {"bbox": [float(x) for x in dict_value["bbox"].to_list()], "text": dict_value["text"]}
            for dict_value in result
        ]

        # Cache the result
        self.cache_manager.set(
            image, self._model_name, "get_boxes_and_text", kwargs, self._package_names, jsonable_result
        )

        return result

    @timetrace("OCRPageEx")
    def extract_page(self, page: Optional[Union["PDFPage", "Image.Image"]]) -> list[Element]:
        assert isinstance(page, Image.Image)
        ocr_output = self.get_boxes_and_text(page)
        width, height = page.size
        return self.parse_output(ocr_output, width, height)

    @timetrace("OCRDocEx")
    def extract_document(self, filename: Union[str, IOBase], **kwargs) -> list[list[Element]]:
        # Note: This method still uses the old document-level caching for backward compatibility
        # The new per-image caching is handled in get_text and get_boxes_and_text methods
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
            return pages


class EasyOcr(OcrModel):
    @requires_modules("easyocr", extra="local-inference")
    def __init__(
        self,
        lang_list=["en"],
        cache_path: Optional[str] = None,
        cache_only: bool = False,
        disable_caching: bool = True,
        **kwargs,
    ):
        super().__init__(cache_path=cache_path, cache_only=cache_only, disable_caching=disable_caching)
        import easyocr

        if os.environ.get("ARYN_AIRGAPPED", "false") == "true":
            md = self._model_dir()
            assert md is not None, "Unable to find pre-downloaded model directory"
            kwargs["model_storage_directory"] = md
            kwargs["download_enabled"] = False

        self.reader = easyocr.Reader(lang_list=lang_list, **kwargs)
        self._lang_list = lang_list
        self._reader_kwargs = kwargs

    def _get_package_names(self) -> list[str]:
        return ["easyocr"]

    def _model_dir(self):
        possibles = ["/aryn/models/easyocr", "/app/models/easyocr"]
        for p in possibles:
            if os.path.exists(p):
                return p + "/"
        return None

    def _get_text_impl(self, image: Image.Image, **kwargs) -> tuple[str, Optional[float]]:
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

    def _get_boxes_and_text_impl(self, image: Image.Image, **kwargs) -> list[dict[str, Any]]:
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
    def __init__(self, cache_path: Optional[str] = None, cache_only: bool = False, disable_caching: bool = True):
        super().__init__(cache_path=cache_path, cache_only=cache_only, disable_caching=disable_caching)
        import pytesseract

        self.pytesseract = pytesseract

    def _get_package_names(self) -> list[str]:
        return ["pytesseract"]

    def _get_text_impl(self, image: Image.Image, **kwargs) -> tuple[str, Optional[float]]:
        val = self.pytesseract.image_to_string(image)
        # font size calculation is not supported for tesseract
        return val, None

    def _get_boxes_and_text_impl(self, image: Image.Image, **kwargs) -> list[dict[str, Any]]:
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
    """Legacy behavior using Tesseract for text and EasyOcr for tables."""

    @requires_modules(["easyocr", "pytesseract"], extra="local-inference")
    def __init__(self, cache_path: Optional[str] = None, cache_only: bool = False, disable_caching: bool = True):
        super().__init__(cache_path=cache_path, cache_only=cache_only, disable_caching=disable_caching)
        self.tesseract = Tesseract(cache_path=cache_path, cache_only=cache_only, disable_caching=disable_caching)
        self.easy_ocr = EasyOcr(cache_path=cache_path, cache_only=cache_only, disable_caching=disable_caching)

    def _get_package_names(self) -> list[str]:
        return ["easyocr", "pytesseract"]

    def _get_text_impl(self, image: Image.Image, **kwargs) -> tuple[str, Optional[float]]:
        # font size calculation is not supported for tesseract
        return self.tesseract._get_text_impl(image, **kwargs)

    def _get_boxes_and_text_impl(self, image: Image.Image, **kwargs) -> list[dict[str, Any]]:
        return self.easy_ocr._get_boxes_and_text_impl(image, **kwargs)

    def __name__(self):
        return "LegacyOcr"


class PaddleOcr(OcrModel):
    """
    PaddleOCR is a state-of-the-art OCR model that uses a combination of
    convolutional neural networks and transformer models to extract text from images.
    It is the default OCR model for Sycamore.
    """

    # NOTE: Also requires the installation of paddlepaddle or paddlepaddle-gpu
    # depending on your system
    @requires_modules(["paddleocr", "paddle"], extra="local-inference")
    def __init__(
        self,
        language="en",
        cache_path: Optional[str] = None,
        cache_only: bool = False,
        disable_caching: bool = True,
        **kwargs,
    ):
        super().__init__(cache_path=cache_path, cache_only=cache_only, disable_caching=disable_caching)
        from paddleocr import PaddleOCR
        import paddle

        device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
        self.predictor = PaddleOCR(
            device=device,
            lang=language,
            **kwargs,
        )
        self._language = language
        self._predictor_kwargs = kwargs

    def _get_package_names(self) -> list[str]:
        return ["paddleocr", "paddle"]

    def _get_text_impl(self, image: Image.Image, **kwargs) -> tuple[str, Optional[float]]:
        result = self.predictor.predict(np.array(image))
        if result and (res := result[0]):
            text_values = []
            font_sizes = []
            for value, bbox in zip(res["rec_texts"], res["rec_boxes"]):
                text_values.append(value)
                font_sizes.append(bbox[3] - bbox[1])
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else None
            return " ".join(text_values), avg_font_size
        return "", None

    def _get_boxes_and_text_impl(
        self, image: Image.Image, get_confidences: bool = False, **kwargs
    ) -> list[dict[str, Any]]:
        result = self.predictor.predict(np.array(image))
        out: list[dict[str, Any]] = []
        if not result or not (res := result[0]):
            return out
        for text, bbox, confidence in zip(res["rec_texts"], res["rec_boxes"], res["rec_scores"]):
            out_dict = {
                "bbox": BoundingBox(*bbox),
                "text": text,
            }
            if get_confidences:
                out_dict["confidence"] = confidence
            out.append(out_dict)
        return out

    def __name__(self):
        return "PaddleOcr"
