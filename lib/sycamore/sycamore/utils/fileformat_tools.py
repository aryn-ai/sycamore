import subprocess
from pathlib import Path
from urllib.parse import urlparse
from tempfile import NamedTemporaryFile, TemporaryDirectory

from sycamore.data import Document


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
    extension = get_file_extension(doc.properties.get("path", "unknown"))

    with NamedTemporaryFile(suffix=f"{extension}") as temp_file:
        temp_file.write(doc.binary_representation)
        temp_file.flush()

        temp_path = Path(temp_file.name)

        run_libreoffice(temp_path, temp_path.parent)

        with open(f"{temp_path.parent}/{temp_path.stem}.pdf", "rb") as processed_file:
            doc.binary_representation = processed_file.read()
            doc.properties["filetype"] = "application/pdf"

    return doc


def get_file_extension(path: str) -> str:
    parsed_url = urlparse(path)
    if parsed_url.scheme in ("s3", "http", "https"):
        path = parsed_url.path
    extension = Path(path).suffix
    return extension
