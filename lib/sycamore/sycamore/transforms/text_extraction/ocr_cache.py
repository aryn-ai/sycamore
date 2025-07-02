import hashlib
import json
import logging
from typing import Any, Dict, Optional
from PIL import Image
import io
import importlib.metadata
from sycamore.utils.cache import Cache, cache_from_path

logger = logging.getLogger(__name__)


class OcrCacheKeyGenerator:
    """Generates cache keys for OCR operations based on image content, model parameters, and versions."""

    def __init__(self):
        self._version_cache = {}

    def _get_package_version(self, package_name: str) -> str:
        """Get the version of a package, caching the result."""
        if package_name not in self._version_cache:
            try:
                version = importlib.metadata.version(package_name)
                self._version_cache[package_name] = version
            except importlib.metadata.PackageNotFoundError:
                self._version_cache[package_name] = "unknown"
        return self._version_cache[package_name]

    def _image_to_hash(self, image: Image.Image) -> str:
        """Convert PIL image to a hash string."""
        # Convert image to bytes for consistent hashing
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return hashlib.sha256(img_bytes.getvalue()).hexdigest()

    def _kwargs_to_hash(self, kwargs: Dict[str, Any]) -> str:
        """Convert keyword arguments to a hash string."""
        # Sort keys for consistent hashing
        sorted_kwargs = dict(sorted(kwargs.items()))
        kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
        return hashlib.sha256(kwargs_str.encode()).hexdigest()

    def generate_key(
        self, image: Image.Image, model_name: str, function_name: str, kwargs: Dict[str, Any], package_names: list[str]
    ) -> str:
        """
        Generate a cache key for OCR operations.

        Args:
            image: PIL Image to process
            model_name: Name of the OCR model (e.g., "PaddleOcr", "EasyOcr")
            function_name: Name of the function being called (e.g., "get_text", "get_boxes_and_text")
            kwargs: Keyword arguments passed to the OCR function
            package_names: List of package names to include in version hash

        Returns:
            Cache key string
        """
        # Hash the image
        image_hash = self._image_to_hash(image)

        # Hash the kwargs
        kwargs_hash = self._kwargs_to_hash(kwargs)

        # Get versions of relevant packages
        versions = {}
        for package in package_names:
            versions[package] = self._get_package_version(package)

        # Create the cache key components
        key_components = {
            "model": model_name,
            "function": function_name,
            "image_hash": image_hash,
            "kwargs_hash": kwargs_hash,
            "versions": versions,
        }

        # Generate final hash
        key_str = json.dumps(key_components, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


class OcrCacheManager:
    """Manages caching for OCR operations with support for local and S3 caches."""

    def __init__(self, cache_path: Optional[str] = None):
        """
        Initialize the OCR cache manager.

        Args:
            cache_path: Path to cache directory or S3 URL (e.g., "s3://bucket/path")
                       If None, uses default local cache at ~/.sycamore/OcrCache
        """
        self.cache: Optional[Cache] = cache_from_path(cache_path) if cache_path is not None else None
        self.key_generator = OcrCacheKeyGenerator()

        if self.cache is None:
            logger.warning("OCR cache is disabled")

    def get(
        self,
        image: Image.Image,
        model_name: str,
        function_name: str,
        kwargs: Dict[str, Any],
        package_names: list[str],
        cache_only: bool = False,
    ) -> Optional[Any]:
        """
        Get cached result for OCR operation.

        Args:
            image: PIL Image to process
            model_name: Name of the OCR model
            function_name: Name of the function being called
            kwargs: Keyword arguments passed to the OCR function
            package_names: List of package names to include in version hash
            cache_only: If True, raise error when cache miss occurs

        Returns:
            Cached result or None if not found

        Raises:
            CacheMissError: If cache_only=True and result not in cache
        """
        if self.cache is None:
            if cache_only:
                raise CacheMissError("Cache is disabled but cache_only=True")
            return None

        key = self.key_generator.generate_key(image, model_name, function_name, kwargs, package_names)
        result = self.cache.get(key)

        if result is None and cache_only:
            raise CacheMissError(f"Cache miss for key: {key}")

        return result

    def set(
        self,
        image: Image.Image,
        model_name: str,
        function_name: str,
        kwargs: Dict[str, Any],
        package_names: list[str],
        result: Any,
    ) -> None:
        """
        Cache result for OCR operation.

        Args:
            image: PIL Image that was processed
            model_name: Name of the OCR model
            function_name: Name of the function that was called
            kwargs: Keyword arguments that were passed to the OCR function
            package_names: List of package names included in version hash
            result: Result to cache
        """
        if self.cache is None:
            return

        key = self.key_generator.generate_key(image, model_name, function_name, kwargs, package_names)
        self.cache.set(key, result)
        logger.debug(f"Cached OCR result for key: {key}")

    def get_hit_rate(self) -> float:
        """Get the cache hit rate."""
        if self.cache is None:
            return 0.0
        return self.cache.get_hit_rate()


class CacheMissError(Exception):
    """Raised when a cache miss occurs and cache_only=True."""

    pass


# Global cache manager instance
_ocr_cache_manager: Optional[OcrCacheManager] = None


def get_ocr_cache_manager(cache_path: Optional[str] = None) -> OcrCacheManager:
    """
    Get the global OCR cache manager instance.

    Args:
        cache_path: Path to cache directory or S3 URL. Only used on first call.

    Returns:
        OcrCacheManager instance
    """
    global _ocr_cache_manager
    if _ocr_cache_manager is None:
        _ocr_cache_manager = OcrCacheManager(cache_path)
    return _ocr_cache_manager


def set_ocr_cache_path(cache_path: str) -> None:
    """
    Set the cache path for OCR operations.

    Args:
        cache_path: Path to cache directory or S3 URL (e.g., "s3://bucket/path")
    """
    global _ocr_cache_manager
    _ocr_cache_manager = OcrCacheManager(cache_path)
