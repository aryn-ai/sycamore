import pdf2image
from sycamore.transforms.text_extraction.ocr_models import PaddleOcr
import os


class TestPaddleOcr:
    def test_paddle_ocr_on_pdf(self):
        document = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "resources",
                "data",
                "pdfs",
                "Ray_page1.pdf",
            )
        )
        reader = PaddleOcr(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_unwarping=False,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
        )
        with open(document, "rb") as f:
            images = pdf2image.convert_from_bytes(f.read())
            with open("test.txt", "w") as f:
                f.write("change")
            image = images[0]
            output = reader.get_boxes_and_text(image)
            texts = " ".join([o["text"] for o in output])
            with open("test.txt", "w") as f:
                f.write(texts)
            assert "Ray: A Distributed Framework for Emerging AI Applications" in texts
            assert "The next generation of AI applications will continuously" in texts
