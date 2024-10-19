from sycamore.data.document import DocumentPropertyTypes, DocumentSource, Document
from sycamore.connectors.common import compare_docs


def compare_connector_docs(gt_docs: list[Document], returned_docs: list[Document], parent_offset: int = 0):
    assert len(gt_docs) == (len(returned_docs) + parent_offset)
    for doc in gt_docs:
        doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY

    gt_dict = {doc.doc_id: doc for doc in gt_docs}
    returned_dict = {doc.doc_id: doc for doc in returned_docs}

    # Find any unmatched doc_ids
    gt_ids = set(gt_dict.keys())
    returned_ids = set(returned_dict.keys())
    missing_from_returned = gt_ids - returned_ids
    extra_in_returned = returned_ids - gt_ids
    assert len(extra_in_returned) == 0
    # Compare all matched documents
    assert all(compare_docs(gt_dict[doc_id], returned_dict[doc_id]) for doc_id in gt_ids.intersection(returned_ids))
    if missing_from_returned:
        assert len(missing_from_returned) == parent_offset
        for missing_doc_id in missing_from_returned:
            assert (gt_doc := gt_dict.get(missing_doc_id)) and not gt_doc.parent_id  # is a parent document
