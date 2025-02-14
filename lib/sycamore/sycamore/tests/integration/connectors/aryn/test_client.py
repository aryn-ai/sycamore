import os

import pytest

from sycamore.connectors.aryn.client import ArynClient


aryn_endpoint = os.getenv("ARYN_ENDPOINT")


@pytest.mark.skip(reason="For manual testing only")
def test_list_docs():
    aryn_api_key = os.getenv("ARYN_TEST_API_KEY")
    client = ArynClient(aryn_url=f"{aryn_endpoint}", api_key=aryn_api_key)
    docset_id = ""
    docs = client.list_docs(docset_id)
    for doc in docs:
        print(doc)


@pytest.mark.skip(reason="For manual testing only")
def test_get_doc():
    aryn_api_key = os.getenv("ARYN_TEST_API_KEY")
    client = ArynClient(aryn_url=f"{aryn_endpoint}", api_key=aryn_api_key)
    docset_id = ""
    docs = client.list_docs(docset_id)
    for doc in docs:
        print(doc)
        doc = client.get_doc(docset_id, doc)
        print(doc)
