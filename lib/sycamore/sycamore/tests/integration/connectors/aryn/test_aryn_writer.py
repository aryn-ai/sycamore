import os
import uuid

import sycamore
from sycamore.data import Document, mkdocid
from aryn_sdk.client import Client

from sycamore.utils.aryn_config import ArynConfig


def test_update_doc_properties():
    doc_id = mkdocid("f")
    dicts = [
        {
            "doc_id": doc_id,
            "elements": [
                {"text_representation": "text-1"},
                {"text_representation": "text-2"},
            ],
        },
    ]

    docs = [Document(item) for item in dicts]

    context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)

    # Give it a unique name to avoid conflicts
    name = str(uuid.uuid4())
    aryn_config = ArynConfig()
    aryn_url = aryn_config.get_aryn_url()
    api_key = aryn_config.get_aryn_api_key()
    original_docs = (
        context.read.document(docs)
        .write.aryn(
            name=name,
            aryn_url=aryn_url,
            api_key=api_key,
        )
        .take_all()
    )

    client = Client(aryn_url=aryn_url, aryn_api_key=api_key)
    res = client.list_docsets(name_eq=name)
    docset_id = res.curr_page[0].docset_id

    dicts = [
        {
            "doc_id": doc_id,
            "properties": {"entity": {"state", "WA"}},
            "elements": [
                {"type": "Text", "text_representation": "text-1"},
                {"type": "Text", "text_representation": "text-2"},
            ],
        },
    ]

    docs = [Document(item) for item in dicts]

    (
        context.read.document(docs)
        .write.aryn(
            docset_id=docset_id,
            aryn_url=aryn_url,
            api_key=api_key,
            update_keys=["properties"],
        )
    )

    doc = client.get_doc(docset_id=docset_id, doc_id=doc_id).value
    assert doc.properties == {"entity": {"state", "WA"}}, f"Expected properties to be updated, got {doc.properties}"

    # Clean up
