"""Manual program for playing around with opensearch sync against real opensearch"""

import logging
from opensearchpy import OpenSearch

import sycamore
from sycamore.data.document import Document
from sycamore.connectors.opensearch.sync import OpenSearchSync
from sycamore.data.docid import path_to_sha256_docid
from sycamore.materialize_config import MRRNameGroup
from sycamore.connectors.opensearch.opensearch_writer import OpenSearchWriterClientParams, OpenSearchWriterTargetParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], verify_certs=False, use_ssl=True)
if False:
    print(client.indices.get_alias("*"))
    exit(0)

if False:
    client.indices.delete("missing-xx")


if False:
    d = [Document(doc_id=path_to_sha256_docid(str(i)), text_representation=str(i)) for i in range(5)]
    sycamore.init(exec_mode=sycamore.EXEC_LOCAL).read.document(d).materialize(
        {"root": "/tmp/xx", "name": MRRNameGroup}
    ).execute()
    d = [Document(doc_id=path_to_sha256_docid(str(i + 10)), text_representation=str(i + 10)) for i in range(5)]
    sycamore.init(exec_mode=sycamore.EXEC_LOCAL).read.document(d).materialize(
        {"root": "/tmp/yy", "name": MRRNameGroup}
    ).execute()

cp = OpenSearchWriterClientParams(verify_certs=False, ssl_show_warn=False)
tp = OpenSearchWriterTargetParams(index_name="eric_test2")


def fake_splitter(doc):
    children = int(doc.text_representation)
    ret = [doc]
    for i in range(children):
        child_id = f"{doc.doc_id}.{i}"
        d = Document(doc_id=path_to_sha256_docid(child_id), parent_id=doc.doc_id, text_representation=str(child_id))
        ret.append(d)

    return ret


oss = OpenSearchSync([("/tmp/xx", fake_splitter)], cp, tp)
# oss = OpenSearchSync([("/tmp/xx", fake_splitter), ("/tmp/yy", fake_splitter)], cp, tp)

oss.sync()
print(oss.stats)
