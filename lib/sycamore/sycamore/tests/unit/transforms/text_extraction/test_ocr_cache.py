import pytest
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch

from sycamore.transforms.text_extraction.ocr_cache import (
    OcrCacheKeyGenerator,
    OcrCacheManager,
    CacheMissError,
    get_ocr_cache_manager,
    set_ocr_cache_path,
)


class TestOcrCacheKeyGenerator:
    def test_image_to_hash(self):
        generator = OcrCacheKeyGenerator()

        # Create two identical images
        img1 = Image.new("RGB", (100, 100), color="white")
        img2 = Image.new("RGB", (100, 100), color="white")

        hash1 = generator._image_to_hash(img1)
        hash2 = generator._image_to_hash(img2)

        assert hash1 == hash2

        # Create a different image
        img3 = Image.new("RGB", (100, 100), color="black")
        hash3 = generator._image_to_hash(img3)

        assert hash1 != hash3

    def test_kwargs_to_hash(self):
        generator = OcrCacheKeyGenerator()

        kwargs1 = {"lang": "en", "device": "cpu"}
        kwargs2 = {"device": "cpu", "lang": "en"}  # Same keys, different order

        hash1 = generator._kwargs_to_hash(kwargs1)
        hash2 = generator._kwargs_to_hash(kwargs2)

        assert hash1 == hash2

        # Different kwargs
        kwargs3 = {"lang": "fr", "device": "cpu"}
        hash3 = generator._kwargs_to_hash(kwargs3)

        assert hash1 != hash3

    @patch("importlib.metadata.version")
    def test_generate_key(self, mock_version):
        mock_version.return_value = "1.0.0"
        generator = OcrCacheKeyGenerator()

        img = Image.new("RGB", (100, 100), color="white")
        kwargs = {"lang": "en"}
        package_names = ["test_package"]

        key1 = generator.generate_key(img, "TestModel", "get_text", kwargs, package_names)
        key2 = generator.generate_key(img, "TestModel", "get_text", kwargs, package_names)

        # Same inputs should produce same key
        assert key1 == key2

        # Different model should produce different key
        key3 = generator.generate_key(img, "DifferentModel", "get_text", kwargs, package_names)
        assert key1 != key3

    def test_generate_key_with_different_functions(self):
        generator = OcrCacheKeyGenerator()

        img = Image.new("RGB", (100, 100), color="white")
        kwargs = {"lang": "en"}
        package_names = ["test_package"]

        key1 = generator.generate_key(img, "TestModel", "get_text", kwargs, package_names)
        key2 = generator.generate_key(img, "TestModel", "get_boxes_and_text", kwargs, package_names)

        # Different functions should produce different keys
        assert key1 != key2

    def test_generate_key_with_empty_kwargs(self):
        generator = OcrCacheKeyGenerator()

        img = Image.new("RGB", (100, 100), color="white")
        kwargs = {}
        package_names = ["test_package"]

        key = generator.generate_key(img, "TestModel", "get_text", kwargs, package_names)
        assert key is not None
        assert len(key) > 0


class TestOcrCacheManager:
    def test_cache_manager_initialization(self, tmp_path):
        cache_path = str(tmp_path / "ocr_cache")
        manager = OcrCacheManager(cache_path)

        assert manager.cache is not None
        assert manager.key_generator is not None

    def test_cache_get_set(self, tmp_path):
        cache_path = str(tmp_path / "ocr_cache")
        manager = OcrCacheManager(cache_path)

        img = Image.new("RGB", (100, 100), color="white")
        kwargs = {"lang": "en"}
        package_names = ["test_package"]
        result = {"text": "test result"}

        # Initially no cache
        assert manager.get(img, "TestModel", "get_text", kwargs, package_names) is None

        # Set cache
        manager.set(img, "TestModel", "get_text", kwargs, package_names, result)

        # Get cached result
        cached_result = manager.get(img, "TestModel", "get_text", kwargs, package_names)
        assert cached_result == result

    def test_cache_only_mode(self, tmp_path):
        cache_path = str(tmp_path / "ocr_cache")
        manager = OcrCacheManager(cache_path)

        img = Image.new("RGB", (100, 100), color="white")
        kwargs = {"lang": "en"}
        package_names = ["test_package"]

        # Should raise error when cache_only=True and no cache
        with pytest.raises(CacheMissError):
            manager.get(img, "TestModel", "get_text", kwargs, package_names, cache_only=True)

        # Set cache
        result = {"text": "test result"}
        manager.set(img, "TestModel", "get_text", kwargs, package_names, result)

        # Should work when cache exists
        cached_result = manager.get(img, "TestModel", "get_text", kwargs, package_names, cache_only=True)
        assert cached_result == result

    def test_cache_disabled(self):
        manager = OcrCacheManager(None)

        img = Image.new("RGB", (100, 100), color="white")
        kwargs = {"lang": "en"}
        package_names = ["test_package"]

        # Should return None when cache is disabled
        assert manager.get(img, "TestModel", "get_text", kwargs, package_names) is None

        # Should raise error when cache_only=True and cache is disabled
        with pytest.raises(CacheMissError):
            manager.get(img, "TestModel", "get_text", kwargs, package_names, cache_only=True)

    def test_hit_rate(self, tmp_path):
        cache_path = str(tmp_path / "ocr_cache")
        manager = OcrCacheManager(cache_path)

        img = Image.new("RGB", (100, 100), color="white")
        kwargs = {"lang": "en"}
        package_names = ["test_package"]
        result = {"text": "test result"}

        # Initially 0 hit rate
        assert manager.get_hit_rate() == 0.0

        # Miss
        manager.get(img, "TestModel", "get_text", kwargs, package_names)
        assert manager.get_hit_rate() == 0.0

        # Set cache
        manager.set(img, "TestModel", "get_text", kwargs, package_names, result)

        # Hit
        manager.get(img, "TestModel", "get_text", kwargs, package_names)
        assert manager.get_hit_rate() == 0.5  # 1 hit, 2 total accesses

    def test_cache_with_different_parameters(self, tmp_path):
        cache_path = str(tmp_path / "ocr_cache")
        manager = OcrCacheManager(cache_path)

        img = Image.new("RGB", (100, 100), color="white")
        package_names = ["test_package"]

        # Different parameters should create different cache entries
        result1 = {"text": "result1"}
        result2 = {"text": "result2"}

        manager.set(img, "TestModel", "get_text", {"lang": "en"}, package_names, result1)
        manager.set(img, "TestModel", "get_text", {"lang": "fr"}, package_names, result2)

        # Should get different results for different parameters
        cached1 = manager.get(img, "TestModel", "get_text", {"lang": "en"}, package_names)
        cached2 = manager.get(img, "TestModel", "get_text", {"lang": "fr"}, package_names)

        assert cached1 == result1
        assert cached2 == result2
        assert cached1 != cached2


class TestGlobalCacheManager:
    def test_get_ocr_cache_manager(self, tmp_path):
        cache_path = str(tmp_path / "ocr_cache")

        # First call should create new manager
        manager1 = get_ocr_cache_manager(cache_path)
        assert manager1 is not None

        # Second call should return same manager
        manager2 = get_ocr_cache_manager()
        assert manager1 is manager2

    def test_set_ocr_cache_path(self, tmp_path):
        cache_path = str(tmp_path / "ocr_cache")

        # Set new cache path
        set_ocr_cache_path(cache_path)

        # Get manager should use new path
        manager = get_ocr_cache_manager()
        assert manager.cache is not None


class TestCacheMissError:
    def test_cache_miss_error_message(self):
        error = CacheMissError("Test error message")
        assert str(error) == "Test error message"


class TestOcrModelsWithCaching:
    """Test OCR models with different caching configurations."""

    @pytest.fixture
    def test_image(self):
        """Create a simple test image."""
        return Image.new("RGB", (100, 100), color="white")

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield str(Path(temp_dir) / "ocr_cache")

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_paddleocr_initialization(self, mock_requires_modules, temp_cache_dir):
        """Test PaddleOcr initialization with different caching modes."""
        from sycamore.transforms.text_extraction.ocr_models import PaddleOcr

        # Test with default caching
        ocr = PaddleOcr(cache_path=temp_cache_dir)
        assert ocr.cache_manager is not None
        assert not ocr.disable_caching
        assert not ocr.cache_only
        assert ocr._model_name == "PaddleOcr"
        assert ocr._package_names == ["paddleocr", "paddle"]

        # Test with cache disabled
        ocr_disabled = PaddleOcr(cache_path=temp_cache_dir, disable_caching=True)
        assert ocr_disabled.cache_manager is None
        assert ocr_disabled.disable_caching

        # Test with cache only mode
        ocr_cache_only = PaddleOcr(cache_path=temp_cache_dir, cache_only=True)
        assert ocr_cache_only.cache_manager is not None
        assert ocr_cache_only.cache_only
        assert not ocr_cache_only.disable_caching

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_easyocr_initialization(self, mock_requires_modules, temp_cache_dir):
        """Test EasyOcr initialization with different caching modes."""
        from sycamore.transforms.text_extraction.ocr_models import EasyOcr

        # Test with default caching
        ocr = EasyOcr(cache_path=temp_cache_dir)
        assert ocr.cache_manager is not None
        assert not ocr.disable_caching
        assert not ocr.cache_only
        assert ocr._model_name == "EasyOcr"
        assert ocr._package_names == ["easyocr"]

        # Test with cache disabled
        ocr_disabled = EasyOcr(cache_path=temp_cache_dir, disable_caching=True)
        assert ocr_disabled.cache_manager is None
        assert ocr_disabled.disable_caching

        # Test with cache only mode
        ocr_cache_only = EasyOcr(cache_path=temp_cache_dir, cache_only=True)
        assert ocr_cache_only.cache_manager is not None
        assert ocr_cache_only.cache_only
        assert not ocr_cache_only.disable_caching

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_tesseract_initialization(self, mock_requires_modules, temp_cache_dir):
        """Test Tesseract initialization with different caching modes."""
        from sycamore.transforms.text_extraction.ocr_models import Tesseract

        # Test with default caching
        ocr = Tesseract(cache_path=temp_cache_dir)
        assert ocr.cache_manager is not None
        assert not ocr.disable_caching
        assert not ocr.cache_only
        assert ocr._model_name == "Tesseract"
        assert ocr._package_names == ["pytesseract"]

        # Test with cache disabled
        ocr_disabled = Tesseract(cache_path=temp_cache_dir, disable_caching=True)
        assert ocr_disabled.cache_manager is None
        assert ocr_disabled.disable_caching

        # Test with cache only mode
        ocr_cache_only = Tesseract(cache_path=temp_cache_dir, cache_only=True)
        assert ocr_cache_only.cache_manager is not None
        assert ocr_cache_only.cache_only
        assert not ocr_cache_only.disable_caching

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_legacyocr_initialization(self, mock_requires_modules, temp_cache_dir):
        """Test LegacyOcr initialization with different caching modes."""
        from sycamore.transforms.text_extraction.ocr_models import LegacyOcr

        # Test with default caching
        ocr = LegacyOcr(cache_path=temp_cache_dir)
        assert ocr.cache_manager is not None
        assert not ocr.disable_caching
        assert not ocr.cache_only
        assert ocr._model_name == "LegacyOcr"
        assert ocr._package_names == ["easyocr", "pytesseract"]

        # Test with cache disabled
        ocr_disabled = LegacyOcr(cache_path=temp_cache_dir, disable_caching=True)
        assert ocr_disabled.cache_manager is None
        assert ocr_disabled.disable_caching

        # Test with cache only mode
        ocr_cache_only = LegacyOcr(cache_path=temp_cache_dir, cache_only=True)
        assert ocr_cache_only.cache_manager is not None
        assert ocr_cache_only.cache_only
        assert not ocr_cache_only.disable_caching

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_ocr_models_caching_behavior(self, mock_requires_modules, temp_cache_dir, test_image):
        """Test that OCR models properly handle caching behavior."""
        from sycamore.transforms.text_extraction.ocr_models import PaddleOcr

        # Mock the OCR implementation to return predictable results
        with patch.object(PaddleOcr, "_get_text_impl", return_value=("test text", 12.0)):
            with patch.object(PaddleOcr, "_get_boxes_and_text_impl", return_value=[{"text": "test", "bbox": None}]):

                # Test with caching enabled
                ocr = PaddleOcr(cache_path=temp_cache_dir)
                result1 = ocr.get_text(test_image)
                result2 = ocr.get_text(test_image)

                # Results should be identical (cached)
                assert result1 == result2
                assert result1 == ("test text", 12.0)

                # Test with caching disabled
                ocr_disabled = PaddleOcr(cache_path=temp_cache_dir, disable_caching=True)
                result3 = ocr_disabled.get_text(test_image)
                result4 = ocr_disabled.get_text(test_image)

                # Results should be identical (same implementation called)
                assert result3 == result4
                assert result3 == ("test text", 12.0)

                # But cache should not be used
                assert ocr_disabled.cache_manager is None

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_cache_only_mode_behavior(self, mock_requires_modules, temp_cache_dir, test_image):
        """Test cache-only mode behavior with OCR models."""
        from sycamore.transforms.text_extraction.ocr_models import PaddleOcr

        # Mock the OCR implementation
        with patch.object(PaddleOcr, "_get_text_impl", return_value=("test text", 12.0)):

            # First, populate cache with normal mode
            ocr_normal = PaddleOcr(cache_path=temp_cache_dir)
            ocr_normal.get_text(test_image)

            # Then test cache-only mode
            ocr_cache_only = PaddleOcr(cache_path=temp_cache_dir, cache_only=True)

            # Should work with cached data
            result = ocr_cache_only.get_text(test_image)
            assert result == ("test text", 12.0)

            # Should fail with new image
            new_image = Image.new("RGB", (50, 50), color="black")
            with pytest.raises(CacheMissError):
                ocr_cache_only.get_text(new_image)

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_parameter_caching_with_ocr_models(self, mock_requires_modules, temp_cache_dir, test_image):
        """Test that different parameters create different cache entries."""
        from sycamore.transforms.text_extraction.ocr_models import PaddleOcr

        # Mock the OCR implementation
        with patch.object(PaddleOcr, "_get_boxes_and_text_impl") as mock_impl:
            mock_impl.return_value = [{"text": "test", "bbox": None}]

            ocr = PaddleOcr(cache_path=temp_cache_dir)

            # Call with different parameters
            ocr.get_boxes_and_text(test_image, get_confidences=False)
            ocr.get_boxes_and_text(test_image, get_confidences=True)

            # Should call implementation twice (different cache entries)
            assert mock_impl.call_count == 2

            # Call again with same parameters
            ocr.get_boxes_and_text(test_image, get_confidences=False)

            # Should not call implementation again (use cache)
            assert mock_impl.call_count == 2

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_cache_hit_rate_with_ocr_models(self, mock_requires_modules, temp_cache_dir, test_image):
        """Test cache hit rate calculation with OCR models."""
        from sycamore.transforms.text_extraction.ocr_models import PaddleOcr

        # Mock the OCR implementation
        with patch.object(PaddleOcr, "_get_text_impl", return_value=("test text", 12.0)):

            ocr = PaddleOcr(cache_path=temp_cache_dir)

            # Initially 0 hit rate
            assert ocr.cache_manager.get_hit_rate() == 0.0

            # First call - miss
            ocr.get_text(test_image)
            assert ocr.cache_manager.get_hit_rate() == 0.0

            # Second call - hit
            ocr.get_text(test_image)
            assert ocr.cache_manager.get_hit_rate() == 0.5

            # Third call - hit
            ocr.get_text(test_image)
            assert (
                abs(ocr.cache_manager.get_hit_rate() - 0.67) < 0.01
            )  # 2 hits, 3 total (allowing for floating point precision)

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_all_ocr_models_package_names(self, mock_requires_modules):
        """Test that all OCR models have correct package names."""
        from sycamore.transforms.text_extraction.ocr_models import PaddleOcr, EasyOcr, Tesseract, LegacyOcr

        # Test package names for each model
        assert PaddleOcr()._get_package_names() == ["paddleocr", "paddle"]
        assert EasyOcr()._get_package_names() == ["easyocr"]
        assert Tesseract()._get_package_names() == ["pytesseract"]
        assert LegacyOcr()._get_package_names() == ["easyocr", "pytesseract"]

    @patch("sycamore.transforms.text_extraction.ocr_models.requires_modules")
    def test_ocr_models_model_names(self, mock_requires_modules):
        """Test that all OCR models have correct model names."""
        from sycamore.transforms.text_extraction.ocr_models import PaddleOcr, EasyOcr, Tesseract, LegacyOcr

        # Test model names for each model
        assert PaddleOcr()._model_name == "PaddleOcr"
        assert EasyOcr()._model_name == "EasyOcr"
        assert Tesseract()._model_name == "Tesseract"
        assert LegacyOcr()._model_name == "LegacyOcr"
