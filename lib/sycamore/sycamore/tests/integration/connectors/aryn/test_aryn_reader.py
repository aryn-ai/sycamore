import uuid

from aryn_sdk.client import Client
import sycamore
from sycamore.connectors.aryn.ArynReader import DocFilter
from sycamore.data import mkdocid, Document
from sycamore.utils.aryn_config import ArynConfig


def test_aryn_reader(exec_mode):
    total_doc_count = 20
    dicts = [
        {
            "doc_id": mkdocid("f"),
            "elements": [
                {
                    "properties": {"_element_index": 0},
                    "type": "Text",
                    "bbox": (0, 0, 0, 0),
                    "text_representation": f"text-{i}",
                },
            ],
        }
        for i in range(total_doc_count)
    ]

    docs = [Document(item) for item in dicts]

    context = sycamore.init(exec_mode)

    # Give it a unique name to avoid conflicts
    name = str(uuid.uuid4())
    aryn_config = ArynConfig()
    aryn_url = aryn_config.get_aryn_url()
    api_key = aryn_config.get_aryn_api_key()
    aryn_url_base = aryn_url[: -len("/v1/storage")]
    (
        context.read.document(docs).write.aryn(
            name=name,
            aryn_url=aryn_url,
            aryn_api_key=api_key,
        )
    )

    client = Client(aryn_url=aryn_url_base, aryn_api_key=api_key)
    res = client.list_docsets(name_eq=name)
    docset_id = None
    for page in res.iter_page():
        if len(page.value) > 0:
            docset_id = page.value[0].docset_id
            break

    assert docset_id is not None

    ds = context.read.aryn(
        docset_id=docset_id,
        aryn_url=aryn_url,
        aryn_api_key=api_key,
    ).take_all()

    assert len(ds) == total_doc_count

    # Test the filter
    docs10 = [doc.doc_id for doc in ds[0:10]]
    ds = context.read.aryn(
        docset_id=docset_id,
        aryn_url=aryn_url,
        aryn_api_key=api_key,
        doc_filter=DocFilter(doc_ids=docs10),
    ).take_all()

    assert len(ds) == 10

    # Clean up the created docset
    client.delete_docset(docset_id=docset_id)


def test_aryn_reader_with_original_elements(exec_mode):
    from aryn_sdk.client import Client

    aryn_url = "https://test-api.aryn.ai/v1/storage"
    aryn_config = ArynConfig()
    client = Client(aryn_url="https://test-api.aryn.ai", aryn_api_key=aryn_config.get_aryn_api_key())
    docset_id = None
    try:
        docset_id = client.create_docset(name=f"test-{str(uuid.uuid4())}").value.docset_id
        print(f"Created docset {docset_id}")

        input = "s3://aryn-public/ntsb_reports/194359.pdf"
        task = client.add_doc_async(docset_id=docset_id, file=input, options={"table_mode": "vision"})
        print(f"Add_doc task ID: {task.task_id}")
        print(f"Add_doc done: {task.result().status_code}")

        assert task.result().status_code == 200

        doc_id = task.result().value.doc_id
        print(f"Added doc ID: {doc_id}")

        doc = client.get_doc(docset_id=docset_id, doc_id=doc_id, include_original_elements=True).value
        before_chunking = len(doc.properties["_original_elements"])
        after_chunking = len(doc.elements)

        context = sycamore.init(exec_mode)
        ds = context.read.aryn(
            docset_id=docset_id,
            aryn_url=aryn_url,
            aryn_api_key=aryn_config.get_aryn_api_key(),
        ).take_all()

        assert len(ds[0].elements) == after_chunking

        ds = context.read.aryn(
            docset_id=docset_id,
            aryn_url=aryn_url,
            aryn_api_key=aryn_config.get_aryn_api_key(),
            use_original_elements=True,
        ).take_all()

        assert len(ds[0].elements) == before_chunking

    finally:
        if docset_id is not None:
            client.delete_docset(docset_id=docset_id)
            print(f"Deleted docset {docset_id}")
