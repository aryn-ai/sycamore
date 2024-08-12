import os.path
import subprocess
from tempfile import NamedTemporaryFile

from sycamore.data import Document


def convert_file_to_pdf(doc: Document) -> Document:
    """
    Utility to convert binary_representations into different file formats. Uses LibreOffice as the conversion engine.

    Note: LibreOffice currently requires manual installation based on your platform.
    """

    def run_libreoffice(source_path, output_path):
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                source_path,
                "--outdir",
                output_path,
            ]
        )

    assert doc.binary_representation is not None

    current_filetype = doc.properties.get("filetype")
    assert current_filetype is not None, "Document requires properties.filetype"

    with NamedTemporaryFile(suffix=f".{current_filetype.split('/')[-1]}") as temp_file:
        filename = ".".join(
            temp_file.name.split("/")[-1].split(".")[:-1]
        )  # standard libs act weird with non-standard formats so we need this
        temp_file.write(doc.binary_representation)
        temp_file.flush()
        processed_path = temp_file.name.split(".")[0] + "_processed"
        run_libreoffice(temp_file.name, processed_path)

        with open(f"{processed_path}/{filename}.pdf", "rb") as processed_file:
            doc.binary_representation = processed_file.read()

    return doc
