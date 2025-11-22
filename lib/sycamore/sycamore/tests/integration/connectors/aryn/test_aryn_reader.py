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
