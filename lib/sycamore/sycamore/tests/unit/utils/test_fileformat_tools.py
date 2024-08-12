import logging
import os
from io import BytesIO

from pypdf import PdfReader

import sycamore
from sycamore.utils.fileformat_tools import convert_file_to_pdf
from sycamore.tests.config import TEST_DIR


def test_pdf_to_pdf():
    # Run this test locally only if libreoffice is installed
    if not os.path.exists("/usr/bin/libreofffice"):
        assert "GITHUB_ACTIONS" not in os.environ
        logging.warning("Skipping test ...; /usr/bin/libreoffice is not installed")
        return
    paths = str(TEST_DIR / "resources/data/docx/aryn_website_sample.docx")

    context = sycamore.init()
    doc = context.read.binary(paths, binary_format="docx").take(1)[0]
    result = convert_file_to_pdf(doc)

    pdf_bytes = BytesIO(result.binary_representation)
    reader = PdfReader(pdf_bytes)
    assert len(reader.pages) == 2
