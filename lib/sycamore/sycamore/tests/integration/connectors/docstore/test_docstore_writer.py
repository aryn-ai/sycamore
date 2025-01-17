import os

import requests
from requests import Response

api_key = os.getenv("ARYN_API_KEY")


def test_write():
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response: Response = requests.post("http://0.0.0.0:8001/v1/docstore/docsets/write",
                                       stream=True, headers=headers)