import os
from PIL import Image
import pytest
from unittest.mock import MagicMock, patch

from sycamore.data import BoundingBox
from sycamore.transforms.text_extraction.ocr_models import EasyOcr, PaddleOcr


@patch.dict(os.environ, {"ARYN_AIRGAPPED": "true"})
def test_air_gap():
    if os.path.exists("/app/models") or os.path.exists("/aryn/models"):
        print("One of /app/models and /aryn/models exists; unable to test airgapping")
        return

    # fails because the model doesn't exist.
    with pytest.raises(AssertionError):
        EasyOcr()


def test_paddle_ocr_output_transform():
    # A sample of the output from paddleocr.tool.predict(img=...)
    mock_data = {
        "rec_texts": [
            "Table of Contents",
            "49",
        ],
        "rec_scores": [
            0.9984432458877563,
            0.9996973872184753,
        ],
        "rec_boxes": [
            [622, 220, 1253, 310],
            [3247, 5874, 3358, 5969],
        ],
    }

    # The actual output of predict is a list of tuples (text, confidence)
    # The result.json has these separated. We zip them for the test.
    mock_output = mock_data.copy()

    # We are not testing the init, so we can disable the check for paddleocr
    with patch.dict("sys.modules", {"paddleocr": MagicMock(), "paddle": MagicMock()}):
        ocr = PaddleOcr()

    ocr.predictor = MagicMock()
    # The method returns a list containing the result dictionary
    ocr.predictor.predict.return_value = [mock_output]

    # The image is not used due to mocking
    image = Image.new("RGB", (100, 100))
    result = ocr.get_boxes_and_text(image)

    assert len(result) == len(mock_output["rec_texts"])

    # check first and last element
    expected_text = mock_output["rec_texts"]
    expected_bbox = BoundingBox(*mock_output["rec_boxes"][0])
    assert result[0]["text"] == expected_text[0]
    assert result[0]["bbox"] == expected_bbox

    expected_text = mock_output["rec_texts"]
    expected_bbox = BoundingBox(*mock_output["rec_boxes"][-1])
    assert result[-1]["text"] == expected_text[-1]
    assert result[-1]["bbox"] == expected_bbox
