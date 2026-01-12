#!/usr/bin/env python
"""
Generate ground truth data for unit test fakes.

This script runs real ML model inference on test files and saves the results
as JSON files that can be used by FakeDeformableDetr and FakeTableStructureExtractor
in unit tests.

Usage:
    # Generate all ground truth
    python scripts/generate_ground_truth.py --all

    # Generate only DETR ground truth
    python scripts/generate_ground_truth.py --detr-only

    # Generate only table structure ground truth
    python scripts/generate_ground_truth.py --table-only

    # Generate for specific PDF file
    python scripts/generate_ground_truth.py --file path/to/file.pdf

    # Generate for specific image file
    python scripts/generate_ground_truth.py --image path/to/image.png

    # List what would be generated without actually generating
    python scripts/generate_ground_truth.py --dry-run --all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

import typing

if typing.TYPE_CHECKING:
    from sycamore.data import TableElement

# Add the sycamore package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sycamore.tests.config import TEST_DIR
from sycamore.tests.unit.transforms import partitioner_fakes

# Ground truth output directory
GROUND_TRUTH_DIR = TEST_DIR / "resources" / "ground_truth"
GROUND_TRUTH_VERSION = "v1"

# Default test resource locations
PDF_DIR = TEST_DIR / "resources" / "data" / "pdfs"
IMG_DIR = TEST_DIR / "resources" / "data" / "imgs"


def generate_detr_ground_truth(
    images: list[tuple[str, Image.Image]],
    output_dir: Path,
    dry_run: bool = False,
) -> list[Path]:
    """
    Generate DETR ground truth for a list of images.

    Args:
        images: List of (source_name, PIL.Image) tuples
        output_dir: Directory to write ground truth files
        dry_run: If True, don't actually write files

    Returns:
        List of paths to generated files
    """
    from sycamore.transforms.detr_partitioner import DeformableDetr, ARYN_DETR_MODEL, _VERSION

    if not dry_run:
        print(f"Loading DETR model: {ARYN_DETR_MODEL}")
        model = DeformableDetr(ARYN_DETR_MODEL)
    else:
        model = None

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for source_name, image in images:
        image_hash, image_meta = partitioner_fakes.image_fingerprint(image)
        output_path = output_dir / f"{image_hash}.json"

        if dry_run:
            print(f"Would generate: {output_path}")
            print(f"  Source: {source_name}")
            print(f"  Image hash: {image_hash}")
            print(f"  Image size: {image_meta['image_size']}")
            generated_files.append(output_path)
            continue

        print(f"Processing: {source_name}")
        print(f"  Image hash: {image_hash}")

        # Run inference with threshold=0 to get all detections
        # Threshold filtering happens in infer() when consuming the ground truth
        assert model is not None

        results = model._get_uncached_inference([image], threshold=0.0)
        result = results[0]

        ground_truth = {
            "version": GROUND_TRUTH_VERSION,
            "model_version": _VERSION,
            "model_name": ARYN_DETR_MODEL,
            "source": source_name,
            "image_hash": image_hash,
            "image_size": image_meta["image_size"],
            "image_mode": image_meta["image_mode"],
            "hash_method": image_meta["hash_method"],
            "hash_length": image_meta["hash_length"],
            "generated_at": datetime.utcnow().isoformat(),
            "results": {
                "scores": result["scores"],
                "labels": result["labels"],
                "boxes": result["boxes"],
            },
        }

        with open(output_path, "w") as f:
            json.dump(ground_truth, f, indent=2)

        print(f"  Wrote: {output_path}")
        generated_files.append(output_path)

    return generated_files


def generate_table_ground_truth(
    tables: list[tuple[str, "TableElement", Image.Image]],
    output_dir: Path,
    dry_run: bool = False,
) -> list[Path]:
    """
    Generate table structure ground truth.

    Args:
        tables: List of (source_name, TableElement, doc_image) tuples
        output_dir: Directory to write ground truth files
        dry_run: If True, don't actually write files

    Returns:
        List of paths to generated files
    """
    from sycamore.transforms.table_structure.extract import TableTransformerStructureExtractor

    if not dry_run:
        print("Loading table structure extractor model")
        extractor = TableTransformerStructureExtractor()
    else:
        extractor = None

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for source_name, element, doc_image in tables:
        if element.bbox is None:
            print(f"Skipping {source_name}: no bounding box")
            continue

        element_hash, table_meta = partitioner_fakes.table_fingerprint(element, doc_image)

        output_path = output_dir / f"{element_hash}.json"

        if dry_run:
            print(f"Would generate: {output_path}")
            print(f"  Source: {source_name}")
            print(f"  Element hash: {element_hash}")
            generated_files.append(output_path)
            continue

        print(f"Processing table: {source_name}")
        print(f"  Element hash: {element_hash}")

        assert extractor is not None

        # Run extraction
        extracted = extractor.extract(element, doc_image)

        ground_truth = {
            "version": GROUND_TRUTH_VERSION,
            "model_name": extractor.model,
            "source": source_name,
            "element_hash": element_hash,
            "bbox": table_meta["bbox"],
            "padding": table_meta["padding"],
            "page_image_size": table_meta["image_size"],
            "page_image_mode": table_meta["image_mode"],
            "hash_method": table_meta["hash_method"],
            "hash_length": table_meta["hash_length"],
            "generated_at": datetime.utcnow().isoformat(),
            "table": extracted.table.to_dict() if extracted.table else None,
        }

        with open(output_path, "w") as f:
            json.dump(ground_truth, f, indent=2)

        print(f"  Wrote: {output_path}")
        generated_files.append(output_path)

    return generated_files


def collect_images_from_pdfs(pdf_paths: list[Path]) -> list[tuple[str, Image.Image]]:
    """Convert PDFs to images for DETR processing."""
    from sycamore.utils.pdf import convert_from_path_streamed_batched

    images = []
    for pdf_path in pdf_paths:
        print(f"Converting PDF to images: {pdf_path}")
        for page_idx, batch in enumerate(convert_from_path_streamed_batched(str(pdf_path), batch_size=1)):
            for image in batch:
                source_name = f"{pdf_path.name}:page{page_idx}"
                images.append((source_name, partitioner_fakes._canonicalize_image(image)))
    return images


def collect_images_from_files(image_paths: list[Path]) -> list[tuple[str, Image.Image]]:
    """Load image files directly."""
    images = []
    for path in image_paths:
        print(f"Loading image: {path}")
        image = Image.open(path)
        images.append((path.name, partitioner_fakes._canonicalize_image(image)))
    return images


def collect_tables_from_pdfs(
    pdf_paths: list[Path] | None = None,
    dry_run: bool = False,
) -> list[tuple[str, "TableElement", Image.Image]]:
    """
    Collect table elements from PDFs by running DETR to identify them.

    Uses FakeDeformableDetr (with existing ground truth) to identify table elements,
    then returns them for table structure extraction.

    Args:
        pdf_paths: List of PDF files to process. If None, uses all PDFs in test resources.
        dry_run: If True, just print what would be processed.

    Returns:
        List of (source_name, TableElement, doc_image) tuples
    """
    from sycamore.utils.pdf import convert_from_path_streamed_batched
    from sycamore.tests.unit.transforms.partitioner_fakes import FakeDeformableDetr
    from sycamore.data import TableElement

    if pdf_paths is None:
        pdf_paths = sorted(PDF_DIR.glob("*.pdf"))

    print(f"Scanning {len(pdf_paths)} PDFs for tables...")

    # Use FakeDeformableDetr which reads from existing ground truth
    detr = FakeDeformableDetr()
    tables = []

    for pdf_path in pdf_paths:
        print(f"Scanning: {pdf_path.name}")

        for page_idx, batch in enumerate(convert_from_path_streamed_batched(str(pdf_path), batch_size=1)):
            for image in batch:
                source_name = f"{pdf_path.name}:page{page_idx}"
                image = partitioner_fakes._canonicalize_image(image)

                if dry_run:
                    print(f"  Would scan {source_name} for tables")
                    continue

                try:
                    # Run DETR to get elements (uses ground truth via fake)
                    elements = detr.infer([image], threshold=0.35)

                    # Find table elements (type is lowercase "table")
                    for elem in elements[0]:
                        if elem.type and elem.type.lower() == "table":
                            # Convert to TableElement
                            table_elem = TableElement(elem.data)
                            tables.append((source_name, table_elem, image))
                            print(f"  Found table in {source_name}")

                except Exception as e:
                    print(f"  Warning: Could not process {source_name}: {e}")
                    continue

    print(f"Found {len(tables)} tables total")
    return tables


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth data for unit test fakes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all", action="store_true", help="Generate all ground truth")
    parser.add_argument("--detr-only", action="store_true", help="Generate only DETR ground truth")
    parser.add_argument("--table-only", action="store_true", help="Generate only table structure ground truth")
    parser.add_argument("--file", type=Path, action="append", help="Process specific PDF file(s)")
    parser.add_argument("--image", type=Path, action="append", help="Process specific image file(s)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without writing files")
    args = parser.parse_args()

    # Validate arguments
    if not any([args.all, args.detr_only, args.table_only, args.file, args.image]):
        parser.print_help()
        print("\nError: Must specify at least one of --all, --detr-only, --table-only, --file, or --image")
        sys.exit(1)

    generate_detr = args.all or args.detr_only or args.file or args.image
    generate_table = args.all or args.table_only

    # Collect images to process
    images: list[tuple[str, Image.Image]] = []

    if args.file:
        images.extend(collect_images_from_pdfs(args.file))

    if args.image:
        images.extend(collect_images_from_files(args.image))

    if args.all or (args.detr_only and not args.file and not args.image):
        # Process all test PDFs and images
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))
        img_files = sorted(IMG_DIR.glob("*.png")) + sorted(IMG_DIR.glob("*.jpg"))

        print(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")
        print(f"Found {len(img_files)} image files in {IMG_DIR}")

        images.extend(collect_images_from_pdfs(pdf_files))
        images.extend(collect_images_from_files(img_files))

    # Generate DETR ground truth
    if generate_detr and images:
        detr_output_dir = GROUND_TRUTH_DIR / "detr" / GROUND_TRUTH_VERSION
        print(f"\n{'=' * 60}")
        print(f"Generating DETR ground truth ({len(images)} images)")
        print(f"Output directory: {detr_output_dir}")
        print(f"{'=' * 60}\n")

        generated = generate_detr_ground_truth(images, detr_output_dir, dry_run=args.dry_run)
        print(f"\nGenerated {len(generated)} DETR ground truth files")

    # Generate table structure ground truth
    # Note: This requires first running DETR to identify table elements
    if generate_table:
        print(f"\n{'=' * 60}")
        print("Table structure ground truth generation")
        print("Note: Uses DETR ground truth to identify tables")
        print(f"{'=' * 60}\n")

        # Collect tables from PDFs using DETR (fake, using existing ground truth)
        tables = collect_tables_from_pdfs(args.file if args.file else None, dry_run=args.dry_run)

        if tables:
            table_output_dir = GROUND_TRUTH_DIR / "table_structure" / GROUND_TRUTH_VERSION
            generated = generate_table_ground_truth(tables, table_output_dir, dry_run=args.dry_run)
            print(f"\nGenerated {len(generated)} table structure ground truth files")
        else:
            print("No tables found in test PDFs")

    print("\nDone!")


if __name__ == "__main__":
    main()
