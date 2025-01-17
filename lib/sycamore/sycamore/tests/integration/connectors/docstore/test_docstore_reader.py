import json
import os

import requests
from requests import Response

import sycamore
from sycamore import ExecMode

api_key = os.getenv("ARYN_API_KEY")

def test_read_all():
    headers  = {
        "Authorization": f"Bearer {api_key}"
    }
    response: Response = requests.post("http://0.0.0.0:8001/v1/docstore/docsets/aryn:test_opensearch_read_large/read", stream=True, headers=headers)
    assert response.status_code == 200
    i = 1
    docs = []
    for chunk in response.iter_lines():
        # print(f"\n{chunk}\n")
        doc = json.loads(chunk)
        docs.append(doc)
        print(f"Doc: {i}")
        i += 1
        print(json.dumps(doc, indent=4))

    print(f"Total docs: {i}")

def test_reader():
    docset_id = "aryn:ds-fw7e13dag460le0sm5hra9i"
    context = sycamore.init(exec_mode=ExecMode.RAY)
    context.read.docstore(docset_id="aryn:test_opensearch_read_large").write.docstore(docset_id)
    # print(f"Total docs: {len(ds)}")
    # for doc in ds:
    # print(ds[0])

    """
    headers  = {
        "Authorization": f"Bearer {api_key}"
    }
    response: Response = requests.post(f"http://0.0.0.0:8001/v1/docstore/docsets/write",
                                       data=ds[0].serialize(),
                                       params={"docset_id": docset_id}, headers=headers)

    print(response.reason)
    """