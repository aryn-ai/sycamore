import os
from sycamore.tests.integration.connectors.common import compare_connector_docs


def test_duckdb_read(shared_ctx, embedded_transformer_paper):
    table_name = "duckdb_table"
    db_url = "tmp_read.db"

    docs = embedded_transformer_paper.take_all()
    shared_ctx.read.document(docs).write.duckdb(db_url=db_url, table_name=table_name, dimensions=384)
    target_doc_id = docs[-1].doc_id if docs[-1].doc_id else ""
    out_docs = shared_ctx.read.duckdb(db_url=db_url, table_name=table_name).take_all()
    query = f"SELECT * from {table_name} WHERE doc_id == '{target_doc_id}'"
    query_docs = shared_ctx.read.duckdb(db_url=db_url, table_name=table_name, query=query).take_all()
    try:
        os.unlink(db_url)
    except Exception as e:
        print(f"Error deleting {db_url}: {e}")
    assert len(query_docs) == 1  # exactly one doc should be returned
    compare_connector_docs(docs, out_docs)
