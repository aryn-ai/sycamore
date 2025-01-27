import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from tempfile import NamedTemporaryFile, TemporaryDirectory

from sycamore.data import Document

logger = logging.getLogger(__name__)


def binary_representation_to_pdf(doc: Document) -> Document:
    """
    Utility to convert binary_representations into different file formats. Uses LibreOffice as the conversion engine.

    Note: LibreOffice currently requires manual installation based on your platform.
    """

    def run_libreoffice(source_path, output_path):
        with TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    source_path,
                    "--outdir",
                    output_path,
                    f"-env:UserInstallation=file://{temp_dir}",
                ]
            )

    assert doc.binary_representation is not None
    origpath = doc.properties.get("path", "unknown")
    extension = get_file_extension(origpath)

    with NamedTemporaryFile(suffix=f"{extension}") as temp_file:
        temp_file.write(doc.binary_representation)
        temp_file.flush()

        temp_path = Path(temp_file.name)

        pdffile = f"{temp_path.parent}/{temp_path.stem}.pdf"
        logger.info(f"Processing {origpath} to {pdffile}")
        with open(pdffile + "-path", "w") as pathfile:
            pathfile.write(origpath)

        run_libreoffice(temp_path, temp_path.parent)

        with open(pdffile, "rb") as processed_file:
            doc.binary_representation = processed_file.read()
            doc.properties["filetype"] = "application/pdf"
        os.unlink(pdffile)
        os.unlink(pdffile + "-path")

    return doc


def get_file_extension(path: str) -> str:
    parsed_url = urlparse(path)
    if parsed_url.scheme in ("s3", "http", "https"):
        path = parsed_url.path
    extension = Path(path).suffix
    return extension
