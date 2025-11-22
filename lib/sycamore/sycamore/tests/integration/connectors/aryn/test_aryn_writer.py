import uuid

import sycamore
from sycamore.data import Document, mkdocid
from aryn_sdk.client import Client

from sycamore.utils.aryn_config import ArynConfig


def test_update_doc_properties(exec_mode):
    doc_id = mkdocid("f")
    dicts = [
        {
            "doc_id": doc_id,
            "elements": [
                {
                    "properties": {"_element_index": 1},
                    "type": "Text",
                    "bbox": (0, 0, 0, 0),
                    "text_representation": "text-1",
                },
                {
                    "properties": {"_element_index": 2},
                    "type": "Text",
                    "bbox": (0, 0, 0, 0),
                    "text_representation": "text-2",
                },
            ],
        },
    ]

    docs = [Document(item) for item in dicts]

    context = sycamore.init(exec_mode=exec_mode)

    # Give it a unique name to avoid conflicts
    name = str(uuid.uuid4())
    aryn_config = ArynConfig()
    aryn_url = aryn_config.get_aryn_url()
    api_key = aryn_config.get_aryn_api_key()
    (
        context.read.document(docs).write.aryn(
            name=name,
            aryn_url=aryn_url,
            aryn_api_key=api_key,
        )
    )

    aryn_url_base = aryn_url[: -len("/v1/storage")]
    client = Client(aryn_url=aryn_url_base, aryn_api_key=api_key)
    res = client.list_docsets(name_eq=name)
    docset_id = None
    for page in res.iter_page():
        if len(page.value) > 0:
            docset_id = page.value[0].docset_id
            break

    assert docset_id is not None

    res = client.list_docs(docset_id=docset_id)
    for page in res.iter_page():
        if len(page.value) > 0:
            doc_id = page.value[0].doc_id
            break

    # Add a new property to the document
    properties = {"entity": {"state": "WA"}}
    dicts = [
        {
            "doc_id": doc_id,
            "properties": properties,
            "elements": [
                {
                    "properties": {"_element_index": 1},
                    "type": "Text",
                    "bbox": (0, 0, 0, 0),
                    "text_representation": "text-1",
                },
                {
                    "properties": {"_element_index": 2},
                    "type": "Text",
                    "bbox": (0, 0, 0, 0),
                    "text_representation": "text-2",
                },
            ],
        },
    ]

    docs = [Document(item) for item in dicts]

    (
        context.read.document(docs).write.aryn(
            docset_id=docset_id,
            aryn_url=aryn_url,
            aryn_api_key=api_key,
            only_properties=True,
        )
    )

    res = client.list_docs(docset_id=docset_id)
    for page in res.iter_page():
        if len(page.value) > 0:
            doc_id = page.value[0].doc_id
            break

    doc = client.get_doc(docset_id=docset_id, doc_id=doc_id).value
    print(doc)

    assert "entity" in doc.properties, "Expected 'entity' property to be present"
    assert (
        doc.properties["entity"] == properties["entity"]
    ), f"Expected properties to be updated, got {doc.properties['entity']}"

    # Override an existing property and also add another property.
    properties = {"entity": {"state": "CA", "city": "San Francisco"}}
    dicts = [
        {
            "doc_id": doc_id,
            "properties": properties,
            "elements": [
                {
                    "properties": {"_element_index": 1},
                    "type": "Text",
                    "bbox": (0, 0, 0, 0),
                    "text_representation": "text-1",
                },
                {
                    "properties": {"_element_index": 2},
                    "type": "Text",
                    "bbox": (0, 0, 0, 0),
                    "text_representation": "text-2",
                },
            ],
        },
    ]

    docs = [Document(item) for item in dicts]

    (
        context.read.document(docs).write.aryn(
            docset_id=docset_id,
            aryn_url=aryn_url,
            aryn_api_key=api_key,
            only_properties=True,
        )
    )

    res = client.list_docs(docset_id=docset_id)
    for page in res.iter_page():
        if len(page.value) > 0:
            doc_id = page.value[0].doc_id
            break

    doc = client.get_doc(docset_id=docset_id, doc_id=doc_id).value
    print(doc)

    assert "entity" in doc.properties, "Expected 'entity' property to be present"
    assert (
        doc.properties["entity"] == properties["entity"]
    ), f"Expected properties to be updated, got {doc.properties['entity']}"

    # Clean up
    client.delete_docset(docset_id=docset_id)
