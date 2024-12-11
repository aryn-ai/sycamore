import time

import sycamore

from sycamore.data import Document


def test_take_stream():
    """
    This test maps a sleep fn on 20 documents, each one doing a sleep for a doc_id * seconds, doc_ids in range(0, 20).
    The test verifies that the time to retrieve the first document is less than the max sleep time.

    If this behavior is performed using a take_all(), even with parallelism > 20, the time to retrieve all documents
    would be at least 19s, because the results wouldn't be returned until all documents are processed,
    the slowest one sleeping for 19 seconds.

    In theory, if your parallelism is "low", streaming could also take longer than 19s (if the slowest doc is processed
    first + ray startup time), we're going to ignore the test in that case because streaming is irrelevant.

    """
    import os

    assert os.cpu_count() >= 2, "This test cannot run on machines with a single CPU"

    num_docs = 20
    docs = []
    for i in range(num_docs):
        docs.append(Document(text_representation=f"Document {i}", doc_id=i, properties={"document_number": i}))

    def delay_doc(document: Document) -> Document:
        time.sleep(document.properties["document_number"])
        return document

    context = sycamore.init()
    docset = context.read.document(docs).map(delay_doc)

    start = time.time()
    for _ in docset.take_stream():
        assert (
            time.time() - start
        ) < num_docs, "Time to first doc should be lesser than num_doc seconds, which is the max delay"
        break
