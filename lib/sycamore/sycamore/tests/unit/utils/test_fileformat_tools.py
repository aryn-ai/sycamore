import logging
import os
import shutil
from io import BytesIO

from pypdf import PdfReader

import sycamore
from sycamore.tests.config import TEST_DIR
from sycamore.utils.fileformat_tools import binary_representation_to_pdf


def test_binary_representation_to_pdf():
    # Run this test locally only if libreoffice is installed
    if shutil.which("libreoffice") is None:
        assert "GITHUB_ACTIONS" not in os.environ
        logging.warning("Skipping test ...; /usr/bin/libreoffice is not installed")
        return
    paths = str(TEST_DIR / "resources/data/docx/aryn_website_sample.docx")

    context = sycamore.init()
    doc = context.read.binary(paths, binary_format="docx").take(1)[0]
    result = binary_representation_to_pdf(doc)

    pdf_bytes = BytesIO(result.binary_representation)
    reader = PdfReader(pdf_bytes)
    assert len(reader.pages) == 2
