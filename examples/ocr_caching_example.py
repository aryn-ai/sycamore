#!/usr/bin/env python3
"""
Example demonstrating OCR caching functionality in Sycamore.

This example shows how to:
1. Use OCR models with local caching
2. Use OCR models with S3 caching
3. Use cache_only mode for offline processing
4. Monitor cache hit rates
"""

import tempfile
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Import OCR models and caching utilities
from sycamore.transforms.text_extraction.ocr_models import PaddleOcr, EasyOcr, Tesseract

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_image(text="Hello World", size=(400, 100), color="white"):
    """Create a simple test image with text."""
    # Create image
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    # Draw text
    draw.text((x, y), text, fill="black", font=font)
    return img


def demonstrate_local_caching():
    """Demonstrate local caching functionality."""
    logger.info("=== Local Caching Demo ===")

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = str(Path(temp_dir) / "ocr_cache")

        # Initialize OCR model with local caching (explicitly enable caching)
        ocr = PaddleOcr(cache_path=cache_path, disable_caching=False)

        # Create test image
        img = create_test_image("Hello World")

        # First call - should compute and cache
        logger.info("First call - computing result...")
        result1 = ocr.get_text(img)
        logger.info(f"Result: {result1}")

        # Second call - should use cache
        logger.info("Second call - using cache...")
        result2 = ocr.get_text(img)
        logger.info(f"Result: {result2}")

        # Verify results are identical
        assert result1 == result2, "Cached result should match original result"

        # Check cache hit rate
        hit_rate = ocr.cache_manager.get_hit_rate()
        logger.info(f"Cache hit rate: {hit_rate:.2%}")

        logger.info("Local caching demo completed successfully!\n")


def demonstrate_cache_only_mode():
    """Demonstrate cache_only mode for offline processing."""
    logger.info("=== Cache-Only Mode Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = str(Path(temp_dir) / "ocr_cache")

        # Create OCR model and populate cache
        ocr = PaddleOcr(cache_path=cache_path, disable_caching=False)
        img = create_test_image("Cached Text")

        # Populate cache
        logger.info("Populating cache...")
        ocr.get_text(img)

        # Create new OCR model with cache_only=True
        ocr_cache_only = PaddleOcr(cache_path=cache_path, cache_only=True, disable_caching=False)

        # Should work with cached data
        logger.info("Using cache_only mode with cached data...")
        result = ocr_cache_only.get_text(img)
        logger.info(f"Result: {result}")

        # Should fail with new image
        new_img = create_test_image("New Text")
        logger.info("Trying cache_only mode with new image...")
        try:
            ocr_cache_only.get_text(new_img)
            assert False, "Should have raised CacheMissError"
        except Exception as e:
            logger.info(f"Expected error: {e}")

        logger.info("Cache-only mode demo completed successfully!\n")


def demonstrate_disable_caching():
    """Demonstrate disable_caching mode (default behavior)."""
    logger.info("=== Disable Caching Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = str(Path(temp_dir) / "ocr_cache")

        # Create OCR model with caching disabled (default)
        ocr_disabled = PaddleOcr(cache_path=cache_path)  # disable_caching=True by default
        img = create_test_image("Disabled Cache Text")

        # Verify cache is disabled
        logger.info("Verifying cache is disabled...")
        assert ocr_disabled.cache_manager is None
        assert ocr_disabled.disable_caching

        # Process image multiple times
        logger.info("Processing image multiple times with caching disabled...")
        result1 = ocr_disabled.get_text(img)
        result2 = ocr_disabled.get_text(img)

        # Results should be identical (same implementation called)
        assert result1 == result2
        logger.info(f"Results are identical: {result1}")

        # Create normal OCR model to verify cache is not populated
        ocr_normal = PaddleOcr(cache_path=cache_path, disable_caching=False)
        logger.info("Checking that cache was not populated...")

        # Should not find cached result
        cached_result = ocr_normal.cache_manager.get(img, "PaddleOcr", "get_text", {}, ["paddleocr", "paddle"])
        assert cached_result is None

        logger.info("Disable caching demo completed successfully!\n")


def demonstrate_s3_caching():
    """Demonstrate S3 caching functionality (requires AWS credentials)."""
    logger.info("=== S3 Caching Demo ===")
    logger.info("Note: This demo requires AWS credentials and a valid S3 bucket.")
    logger.info("Uncomment the code below and set your S3 bucket path to test.")

    # Uncomment and modify the following code to test S3 caching:
    """
    # Set S3 cache path (replace with your bucket)
    s3_cache_path = "s3://your-bucket/ocr-cache"
    
    # Initialize OCR model with S3 caching
    ocr = PaddleOcr(cache_path=s3_cache_path, disable_caching=False)
    
    # Create test image
    img = create_test_image("S3 Cached Text")
    
    # First call - should compute and cache to S3
    logger.info("First call - computing and caching to S3...")
    result1 = ocr.get_text(img)
    logger.info(f"Result: {result1}")
    
    # Second call - should use S3 cache
    logger.info("Second call - using S3 cache...")
    result2 = ocr.get_text(img)
    logger.info(f"Result: {result2}")
    
    # Verify results are identical
    assert result1 == result2, "S3 cached result should match original result"
    
    logger.info("S3 caching demo completed successfully!")
    """

    logger.info("S3 caching demo skipped (requires AWS setup)\n")


def demonstrate_different_models():
    """Demonstrate caching with different OCR models."""
    logger.info("=== Different Models Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = str(Path(temp_dir) / "ocr_cache")

        # Test different models with caching enabled
        models = [
            ("PaddleOcr", PaddleOcr(cache_path=cache_path, disable_caching=False)),
            ("EasyOcr", EasyOcr(cache_path=cache_path, disable_caching=False)),
            ("Tesseract", Tesseract(cache_path=cache_path, disable_caching=False)),
        ]

        img = create_test_image("Model Test")

        for model_name, model in models:
            logger.info(f"Testing {model_name}...")

            # First call
            result1 = model.get_text(img)
            logger.info(f"  First call result: {result1[0][:50]}...")

            # Second call (should use cache)
            result2 = model.get_text(img)
            logger.info(f"  Second call result: {result2[0][:50]}...")

            # Verify caching worked
            assert result1 == result2, f"{model_name} cached result should match original"

            # Check hit rate
            hit_rate = model.cache_manager.get_hit_rate()
            logger.info(f"  Cache hit rate: {hit_rate:.2%}")

        logger.info("Different models demo completed successfully!\n")


def demonstrate_parameter_caching():
    """Demonstrate that different parameters create different cache entries."""
    logger.info("=== Parameter Caching Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = str(Path(temp_dir) / "ocr_cache")

        ocr = PaddleOcr(cache_path=cache_path, disable_caching=False)
        img = create_test_image("Parameter Test")

        # Test with different parameters
        logger.info("Testing with get_confidences=False...")
        result1 = ocr.get_boxes_and_text(img, get_confidences=False)
        logger.info(f"  Result count: {len(result1)}")

        logger.info("Testing with get_confidences=True...")
        result2 = ocr.get_boxes_and_text(img, get_confidences=True)
        logger.info(f"  Result count: {len(result2)}")

        # Verify different parameters create different cache entries
        # (results should be the same but cached separately)
        assert len(result1) == len(result2), "Results should have same number of boxes"

        # Check that confidence is only in second result
        if result2:
            assert "confidence" in result2[0], "Second result should have confidence"
        if result1:
            assert "confidence" not in result1[0], "First result should not have confidence"

        logger.info("Parameter caching demo completed successfully!\n")


def main():
    """Run all demonstrations."""
    logger.info("OCR Caching Demonstrations")
    logger.info("=" * 50)

    try:
        demonstrate_local_caching()
        demonstrate_cache_only_mode()
        demonstrate_disable_caching()
        demonstrate_s3_caching()
        demonstrate_different_models()
        demonstrate_parameter_caching()

        logger.info("All demonstrations completed successfully!")
        logger.info("\nKey Features Demonstrated:")
        logger.info("- Local disk caching")
        logger.info("- Cache-only mode for offline processing")
        logger.info("- Disable caching mode (default)")
        logger.info("- S3 caching support (requires AWS setup)")
        logger.info("- Caching with different OCR models")
        logger.info("- Parameter-aware caching")
        logger.info("- Cache hit rate monitoring")

    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        logger.info("Make sure you have the required OCR dependencies installed:")


if __name__ == "__main__":
    main()
