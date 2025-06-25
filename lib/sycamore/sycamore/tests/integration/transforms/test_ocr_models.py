import pdf2image
import tempfile

from sycamore.transforms.text_extraction.ocr_models import PaddleOcr
from sycamore.tests.config import TEST_DIR


class TestPaddleOcr:
    def test_paddle_ocr_on_pdf(self):
        document = TEST_DIR / "resources" / "data" / "pdfs" / "Transformer.pdf"
        reader = PaddleOcr(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_unwarping=False,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(document, "rb") as f:
                images = pdf2image.convert_from_bytes(f.read(), output_folder=tmp_dir)
                image = images[0]
            output = reader.get_boxes_and_text(image)

            texts = " ".join([o["text"] for o in output])

            # Doesn't work in Paddle 2.0
            assert "the Transformer, based" in texts
            assert "modeling and transduction problems" in texts
