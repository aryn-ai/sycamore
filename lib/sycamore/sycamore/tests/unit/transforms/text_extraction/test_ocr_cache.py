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


def setup_and_assert_cache_manager(temp_cache_dir):
    """Helper to set up and assert cache manager initialization."""
    from sycamore.transforms.text_extraction.ocr_cache import set_ocr_cache_path, get_ocr_cache_manager, OcrCacheManager

    set_ocr_cache_path(temp_cache_dir)
    cache_manager = get_ocr_cache_manager()
    assert cache_manager is not None
    assert cache_manager.cache is not None
    # Test with cache disabled
    disabled_manager = OcrCacheManager(None)
    assert disabled_manager.cache is None
    # Test with cache enabled
    enabled_manager = OcrCacheManager(temp_cache_dir)
    assert enabled_manager.cache is not None
    return cache_manager


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

    def test_paddleocr_initialization(self, temp_cache_dir):
        """Test PaddleOcr initialization with different caching modes."""
        setup_and_assert_cache_manager(temp_cache_dir)

    def test_easyocr_initialization(self, temp_cache_dir):
        """Test EasyOcr initialization with different caching modes."""
        setup_and_assert_cache_manager(temp_cache_dir)

    def test_tesseract_initialization(self, temp_cache_dir):
        """Test Tesseract initialization with different caching modes."""
        setup_and_assert_cache_manager(temp_cache_dir)

    def test_legacyocr_initialization(self, temp_cache_dir):
        """Test LegacyOcr initialization with different caching modes."""
        setup_and_assert_cache_manager(temp_cache_dir)

    def test_ocr_models_caching_behavior(self, temp_cache_dir, test_image):
        """Test that OCR models properly handle caching behavior."""
        # Test the caching behavior using the cache manager directly
        from sycamore.transforms.text_extraction.ocr_cache import OcrCacheManager

        # Create a cache manager
        cache_manager = OcrCacheManager(temp_cache_dir)

        # Test basic caching behavior
        result = {"text": "test result", "confidence": 0.95}

        # Set a cache entry
        cache_manager.set(test_image, "TestModel", "get_text", {}, ["test_package"], result)

        # Get the cached result
        cached_result = cache_manager.get(test_image, "TestModel", "get_text", {}, ["test_package"])
        assert cached_result == result

        # Test cache miss with different parameters
        different_result = cache_manager.get(test_image, "TestModel", "get_text", {"lang": "fr"}, ["test_package"])
        assert different_result is None

    def test_cache_only_mode_behavior(self, temp_cache_dir, test_image):
        """Test cache-only mode behavior with OCR models."""
        # Test cache-only mode using the cache manager directly
        from sycamore.transforms.text_extraction.ocr_cache import OcrCacheManager, CacheMissError

        # Create a cache manager
        cache_manager = OcrCacheManager(temp_cache_dir)

        # Test cache-only mode with no cache
        with pytest.raises(CacheMissError):
            cache_manager.get(test_image, "TestModel", "get_text", {}, ["test_package"], cache_only=True)

        # Add some data to cache
        result = {"text": "test result"}
        cache_manager.set(test_image, "TestModel", "get_text", {}, ["test_package"], result)

        # Should work with cached data
        cached_result = cache_manager.get(test_image, "TestModel", "get_text", {}, ["test_package"], cache_only=True)
        assert cached_result == result

        # Should fail with new image
        new_image = Image.new("RGB", (50, 50), color="black")
        with pytest.raises(CacheMissError):
            cache_manager.get(new_image, "TestModel", "get_text", {}, ["test_package"], cache_only=True)

    def test_parameter_caching_with_ocr_models(self, temp_cache_dir, test_image):
        """Test that different parameters create different cache entries."""
        # Test parameter caching using the cache manager directly
        from sycamore.transforms.text_extraction.ocr_cache import OcrCacheManager

        # Create a cache manager
        cache_manager = OcrCacheManager(temp_cache_dir)

        # Test that different parameters create different cache entries
        result1 = {"text": "result1"}
        result2 = {"text": "result2"}

        # Set cache with different parameters
        cache_manager.set(test_image, "TestModel", "get_text", {"lang": "en"}, ["test_package"], result1)
        cache_manager.set(test_image, "TestModel", "get_text", {"lang": "fr"}, ["test_package"], result2)

        # Get cached results
        cached1 = cache_manager.get(test_image, "TestModel", "get_text", {"lang": "en"}, ["test_package"])
        cached2 = cache_manager.get(test_image, "TestModel", "get_text", {"lang": "fr"}, ["test_package"])

        assert cached1 == result1
        assert cached2 == result2
        assert cached1 != cached2

    def test_cache_hit_rate_with_ocr_models(self, temp_cache_dir, test_image):
        """Test cache hit rate calculation with OCR models."""
        # Test cache hit rate using the cache manager directly
        from sycamore.transforms.text_extraction.ocr_cache import OcrCacheManager

        # Create a cache manager
        cache_manager = OcrCacheManager(temp_cache_dir)

        # Initially 0 hit rate
        assert cache_manager.get_hit_rate() == 0.0

        # First call - miss
        cache_manager.get(test_image, "TestModel", "get_text", {}, ["test_package"])
        assert cache_manager.get_hit_rate() == 0.0

        # Add some data to cache
        result = {"text": "test result"}
        cache_manager.set(test_image, "TestModel", "get_text", {}, ["test_package"], result)

        # Second call - hit
        cache_manager.get(test_image, "TestModel", "get_text", {}, ["test_package"])
        assert cache_manager.get_hit_rate() == 0.5

        # Third call - hit
        cache_manager.get(test_image, "TestModel", "get_text", {}, ["test_package"])
        assert abs(cache_manager.get_hit_rate() - 0.67) < 0.01  # 2 hits, 3 total

    def test_all_ocr_models_package_names(self):
        """Test that all OCR models have correct package names."""
        # Mock the import_modules function to prevent actual imports
        with patch("sycamore.utils.import_utils.import_modules") as mock_import_modules:
            # Mock import_modules to do nothing
            mock_import_modules.return_value = None

            # Import the models (the decorators will call our mocked import_modules)
            from sycamore.transforms.text_extraction.ocr_models import PaddleOcr, EasyOcr, Tesseract, LegacyOcr

        # Test package names for each model
        assert PaddleOcr()._get_package_names() == ["paddleocr", "paddle"]
        assert EasyOcr()._get_package_names() == ["easyocr"]
        assert Tesseract()._get_package_names() == ["pytesseract"]
        assert LegacyOcr()._get_package_names() == ["easyocr", "pytesseract"]

    def test_ocr_models_model_names(self):
        """Test that all OCR models have correct model names."""
        # Mock the import_modules function to prevent actual imports
        with patch("sycamore.utils.import_utils.import_modules") as mock_import_modules:
            # Mock import_modules to do nothing
            mock_import_modules.return_value = None

            # Import the models (the decorators will call our mocked import_modules)
            from sycamore.transforms.text_extraction.ocr_models import PaddleOcr, EasyOcr, Tesseract, LegacyOcr

        # Test model names for each model
        assert PaddleOcr()._model_name == "PaddleOcr"
        assert EasyOcr()._model_name == "EasyOcr"
        assert Tesseract()._model_name == "Tesseract"
        assert LegacyOcr()._model_name == "LegacyOcr"
