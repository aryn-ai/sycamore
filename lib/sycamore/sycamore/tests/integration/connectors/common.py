from sycamore.data.document import DocumentPropertyTypes, DocumentSource
from sycamore.connectors.common import compare_docs


def compare_connector_docs(gt_docs, returned_docs):
    assert len(returned_docs) == len(gt_docs)
    for doc in gt_docs:
        doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
    assert all(
        compare_docs(original, plumbed)
        for original, plumbed in zip(
            sorted(gt_docs, key=lambda d: d.doc_id or ""), sorted(returned_docs, key=lambda d: d.doc_id or "")
        )
    )
