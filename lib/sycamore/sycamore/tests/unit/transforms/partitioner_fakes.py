"""
Fake implementations of ML models for fast unit testing.

These fakes return pre-recorded ground truth data instead of running actual inference.
Ground truth is stored in sycamore/tests/resources/ground_truth/ and can be regenerated
using scripts/generate_ground_truth.py when models change.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

from PIL import Image

from sycamore.data import TableElement, Table
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.detr_partitioner import DeformableDetr, SycamoreObjectDetection, _VERSION
from sycamore.utils.cache import Cache

logger = logging.getLogger(__name__)

# Ground truth version - increment when regenerating ground truth
GROUND_TRUTH_VERSION = "v1"
GROUND_TRUTH_DIR = TEST_DIR / "resources" / "ground_truth"
CANONICAL_IMAGE_MODE = "RGB"
HASH_METHOD = "sha256(rgb_bytes)"
HASH_LENGTH = 16
TABLE_PADDING = 10


class GroundTruthNotFoundError(Exception):
    """Raised when ground truth data is missing for a test input."""

    pass


def _canonicalize_image(image: Image.Image) -> Image.Image:
    """Return an image in the canonical mode used for hashing."""
    if image.mode == CANONICAL_IMAGE_MODE:
        return image
    return image.convert(CANONICAL_IMAGE_MODE)


def image_fingerprint(image: Image.Image) -> Tuple[str, dict[str, Any]]:
    """Compute a stable fingerprint for an image."""
    canonical = _canonicalize_image(image)
    hash_ctx = Cache.get_hash_context(canonical.tobytes())
    digest = hash_ctx.hexdigest()[:HASH_LENGTH]
    metadata = {
        "hash": digest,
        "hash_method": HASH_METHOD,
        "hash_length": HASH_LENGTH,
        "image_mode": canonical.mode,
        "image_size": list(canonical.size),
    }
    return digest, metadata


def table_fingerprint(
    element: TableElement, doc_image: Image.Image, padding: int = TABLE_PADDING
) -> Tuple[str, dict[str, Any]]:
    """Compute a stable fingerprint for a table region."""
    if element.bbox is None:
        raise ValueError("TableElement must have a bounding box")

    canonical_image = _canonicalize_image(doc_image)
    bbox = element.bbox
    bbox_str = f"{bbox.x1:.6f},{bbox.y1:.6f},{bbox.x2:.6f},{bbox.y2:.6f}"
    hash_ctx = Cache.get_hash_context(bbox_str.encode())

    width, height = canonical_image.size
    crop_box = (
        max(int(bbox.x1 * width) - padding, 0),
        max(int(bbox.y1 * height) - padding, 0),
        min(int(bbox.x2 * width) + padding, width),
        min(int(bbox.y2 * height) + padding, height),
    )
    cropped = canonical_image.crop(crop_box).convert(CANONICAL_IMAGE_MODE)
    hash_ctx.update(cropped.tobytes())
    element_hash = hash_ctx.hexdigest()[:HASH_LENGTH]
    metadata = {
        "hash": element_hash,
        "hash_method": HASH_METHOD,
        "hash_length": HASH_LENGTH,
        "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
        "padding": padding,
        "image_mode": canonical_image.mode,
        "image_size": list(canonical_image.size),
    }
    return element_hash, metadata


def _validate_version(gt_data: dict[str, Any], kind: str) -> None:
    """Ensure the stored ground truth version matches the expected version."""
    version = gt_data.get("version")
    if version and version != GROUND_TRUTH_VERSION:
        raise GroundTruthNotFoundError(
            f"{kind} ground truth version mismatch: expected '{GROUND_TRUTH_VERSION}' but found '{version}'. "
            "Regenerate ground truth with 'python scripts/generate_ground_truth.py'."
        )


def _format_candidates(directory: Path) -> str:
    """List nearby ground-truth files to aid debugging."""
    if not directory.exists():
        return "No ground truth directory found"
    files = sorted(p.name for p in directory.glob("*.json"))
    if not files:
        return "No ground truth files in directory"
    return "Available files: " + ", ".join(files)


class FakeDeformableDetr(DeformableDetr):
    """
    Fake DETR model that returns pre-recorded ground truth instead of running inference.

    Inherits from DeformableDetr and only overrides _get_uncached_inference(), so the
    real infer() logic (Element conversion, caching, threshold filtering) is still exercised.

    Ground truth is keyed by image hash and stored in:
        tests/resources/ground_truth/detr/{version}/{image_hash}.json
    """

    GROUND_TRUTH_SUBDIR = "detr"

    def __init__(
        self,
        model_name_or_path: str = "Aryn/deformable-detr-DocLayNet",
        device: Optional[str] = None,
        cache: Optional[Cache] = None,
    ):
        """
        Initialize fake DETR without loading the heavy model weights.

        Sets up the same attributes as the real DeformableDetr but skips loading
        the actual neural network model.
        """
        # Skip DeformableDetr.__init__() which loads the model
        # Call the grandparent's __init__ instead
        SycamoreObjectDetection.__init__(self)

        from sycamore.transforms.detr_partitioner import ARYN_DETR_MODEL

        # Set up attributes that the parent class methods expect
        self._model_name_or_path = model_name_or_path or ARYN_DETR_MODEL
        self.device = device
        self.cache = cache

        # Labels must match the real model
        self.labels = [
            "N/A",
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Section-header",
            "Table",
            "Text",
            "Title",
        ]

        # Model is NOT loaded - we use ground truth instead
        self.model = None
        self.processor = None  # Not needed since we override _get_uncached_inference

        # Cache for ground truth lookups (separate from the inference cache)
        self._ground_truth_cache: dict[str, Any] = {}

    def _hash_image(self, image: Image.Image) -> tuple[str, dict[str, Any]]:
        """Generate a hash and metadata from image bytes for ground truth lookup."""
        return image_fingerprint(image)

    def _get_ground_truth_path(self, image_hash: str) -> Path:
        """Get the path to the ground truth file for an image."""
        return GROUND_TRUTH_DIR / self.GROUND_TRUTH_SUBDIR / GROUND_TRUTH_VERSION / f"{image_hash}.json"

    def _load_ground_truth(self, image_hash: str) -> dict[str, Any]:
        """Load ground truth data for an image hash."""
        if image_hash in self._ground_truth_cache:
            return self._ground_truth_cache[image_hash]

        gt_path = self._get_ground_truth_path(image_hash)
        if not gt_path.exists():
            raise GroundTruthNotFoundError(
                f"Ground truth not found for image hash '{image_hash}'.\n"
                f"Expected file: {gt_path}\n"
                f"Run 'python scripts/generate_ground_truth.py' to generate ground truth data.\n"
                f"{_format_candidates(gt_path.parent)}"
            )

        with open(gt_path) as f:
            data = json.load(f)

        self._ground_truth_cache[image_hash] = data
        return data

    def _validate_ground_truth_metadata(
        self, gt_data: dict[str, Any], expected_meta: dict[str, Any], image_hash: str
    ) -> None:
        """Validate stored metadata before using ground truth."""
        _validate_version(gt_data, "DETR")

        model_version = gt_data.get("model_version")
        if model_version and model_version != _VERSION:
            raise GroundTruthNotFoundError(
                f"Ground truth for hash '{image_hash}' was generated with model_version '{model_version}', "
                f"but current version is '{_VERSION}'. Regenerate ground truth."
            )

        def _check_field(field: str, expected_val: Any) -> None:
            if field in gt_data and gt_data[field] != expected_val:
                raise GroundTruthNotFoundError(
                    f"Ground truth for hash '{image_hash}' has {field}={gt_data[field]!r} "
                    f"but expected {expected_val!r}. Regenerate ground truth."
                )

        _check_field("image_hash", image_hash)
        _check_field("hash_method", expected_meta.get("hash_method"))
        _check_field("hash_length", expected_meta.get("hash_length"))
        if "image_size" in gt_data and "image_size" in expected_meta:
            _check_field("image_size", expected_meta["image_size"])
        if "image_mode" in gt_data:
            _check_field("image_mode", expected_meta.get("image_mode"))

    def _get_uncached_inference(self, images: list[Image.Image], threshold: float) -> list[dict[str, Any]]:
        """
        Return pre-recorded ground truth instead of running model inference.

        This overrides the parent's method which runs the actual neural network.
        The returned format matches what the real model produces:
        list of dicts with 'scores', 'labels', 'boxes'.

        Note: Ground truth is stored WITHOUT threshold filtering (threshold=0), so we
        apply threshold filtering here to match what the real model's post_process_object_detection does.
        """
        results = []
        for image in images:
            image_hash, image_meta = self._hash_image(image)
            gt_data = self._load_ground_truth(image_hash)

            # Validate metadata when available (older ground truth may not have these fields)
            try:
                self._validate_ground_truth_metadata(gt_data, image_meta, image_hash)
            except GroundTruthNotFoundError:
                # Re-raise to keep the helpful message
                raise
            except Exception as e:
                # Avoid breaking older fixtures; warn instead.
                logger.warning("Ground truth metadata validation skipped: %s", e)

            # Filter by threshold (same as post_process_object_detection would do)
            scores = gt_data["results"]["scores"]
            labels = gt_data["results"]["labels"]
            boxes = gt_data["results"]["boxes"]

            filtered_scores = []
            filtered_labels = []
            filtered_boxes = []
            for score, label, box in zip(scores, labels, boxes):
                if score >= threshold:
                    filtered_scores.append(score)
                    filtered_labels.append(label)
                    filtered_boxes.append(box)

            result = {
                "scores": filtered_scores,
                "labels": filtered_labels,
                "boxes": filtered_boxes,
            }
            results.append(result)

        return results


class FakeTableStructureExtractor:
    """
    Fake table structure extractor that returns pre-recorded ground truth.

    Note: Unlike FakeDeformableDetr, this doesn't inherit from TableTransformerStructureExtractor
    because that class's extract() method mixes model inference with post-processing in a way
    that's hard to separate. This fake provides the same interface but returns cached results.

    Ground truth is keyed by a hash of the cropped table image and stored in:
        tests/resources/ground_truth/table_structure/{version}/{element_hash}.json
    """

    GROUND_TRUTH_SUBDIR = "table_structure"

    def __init__(self, model: str = "microsoft/table-structure-recognition-v1.1-all", device: Optional[str] = None):
        """Initialize fake table extractor without loading model weights."""
        self.model = model
        self.device = device
        self.structure_model = None  # Not loaded
        self._ground_truth_cache: dict[str, Any] = {}

    def _hash_table_region(self, element: TableElement, doc_image: Image.Image) -> tuple[str, dict[str, Any]]:
        """Generate hash from table bounding box and cropped image region."""
        return table_fingerprint(element, doc_image)

    def _get_ground_truth_path(self, element_hash: str) -> Path:
        """Get the path to the ground truth file for a table element."""
        return GROUND_TRUTH_DIR / self.GROUND_TRUTH_SUBDIR / GROUND_TRUTH_VERSION / f"{element_hash}.json"

    def _load_ground_truth(self, element_hash: str) -> dict[str, Any]:
        """Load ground truth data for a table element hash."""
        if element_hash in self._ground_truth_cache:
            return self._ground_truth_cache[element_hash]

        gt_path = self._get_ground_truth_path(element_hash)
        if not gt_path.exists():
            raise GroundTruthNotFoundError(
                f"Ground truth not found for table element hash '{element_hash}'.\n"
                f"Expected file: {gt_path}\n"
                f"Run 'python scripts/generate_ground_truth.py' to generate ground truth data.\n"
                f"{_format_candidates(gt_path.parent)}"
            )

        with open(gt_path) as f:
            data = json.load(f)

        self._ground_truth_cache[element_hash] = data
        return data

    def _validate_ground_truth_metadata(
        self, gt_data: dict[str, Any], expected_meta: dict[str, Any], element_hash: str
    ) -> None:
        """Validate stored metadata before using ground truth."""
        _validate_version(gt_data, "Table structure")

        def _check_field(field: str, expected_val: Any) -> None:
            if field in gt_data and gt_data[field] != expected_val:
                raise GroundTruthNotFoundError(
                    f"Ground truth for table hash '{element_hash}' has {field}={gt_data[field]!r} "
                    f"but expected {expected_val!r}. Regenerate ground truth."
                )

        _check_field("element_hash", element_hash)
        _check_field("hash_method", expected_meta.get("hash_method"))
        _check_field("hash_length", expected_meta.get("hash_length"))
        if "bbox" in gt_data and "bbox" in expected_meta:
            _check_field("bbox", expected_meta["bbox"])
        if "padding" in gt_data and "padding" in expected_meta:
            _check_field("padding", expected_meta["padding"])
        if "page_image_size" in gt_data and "image_size" in expected_meta:
            _check_field("page_image_size", expected_meta["image_size"])
        if "page_image_mode" in gt_data and "image_mode" in expected_meta:
            _check_field("page_image_mode", expected_meta["image_mode"])

    def extract(
        self,
        element: TableElement,
        doc_image: Image.Image,
        union_tokens: bool = False,
        apply_thresholds: bool = False,
        resolve_overlaps: bool = False,
    ) -> TableElement:
        """
        Extract table structure using pre-recorded ground truth.

        Returns a TableElement with the table property populated from cached data.
        """
        if element.bbox is None:
            return element

        element_hash, table_meta = self._hash_table_region(element, doc_image)
        gt_data = self._load_ground_truth(element_hash)

        try:
            self._validate_ground_truth_metadata(gt_data, table_meta, element_hash)
        except GroundTruthNotFoundError:
            raise
        except Exception as e:
            logger.warning("Ground truth metadata validation skipped: %s", e)

        # Reconstruct the Table from ground truth
        if gt_data.get("table"):
            element.table = Table.from_dict(gt_data["table"])
        else:
            element.table = None

        return element
