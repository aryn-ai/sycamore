import os

from sycamore.connectors.aryn.client import ArynClient


def test_list_docs():
    aryn_api_key = os.getenv("ARYN_TEST_API_KEY")
    client = ArynClient(aryn_url="http://localhost:8002/v1/docstore", api_key=aryn_api_key)
    docset_id = "aryn:ds-fwaagauoj6yqcia2n4c3zfd"
    docs = client.list_docs(docset_id)
    for doc in docs:
        print(doc)

def test_get_doc():
    aryn_api_key = os.getenv("ARYN_TEST_API_KEY")
    client = ArynClient(aryn_url="http://localhost:8002/v1/docstore", api_key=aryn_api_key)
    docset_id = "aryn:ds-fwaagauoj6yqcia2n4c3zfd"
    docs = client.list_docs(docset_id)
    for doc in docs:
        print(doc)
        doc = client.get_doc(docset_id, doc)
        print(doc)
