import os
import pytest
from sycamore.transforms.text_extraction.ocr_models import EasyOcr
from unittest.mock import patch


@patch.dict(os.environ, {"ARYN_AIRGAPPED": "true"})
def test_air_gap():
    if os.path.exists("/app/models") or os.path.exists("/aryn/models"):
        print("One of /app/models and /aryn/models exists; unable to test airgapping")
        return

    # fails because the model doesn't exist.
    with pytest.raises(AssertionError):
        EasyOcr()
