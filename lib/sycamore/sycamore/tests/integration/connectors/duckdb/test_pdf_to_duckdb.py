import duckdb
import os


def test_to_duckdb(embedded_transformer_paper):
    table_name = "duckdb_table"
    db_url = "tmp_write.db"

    ds = embedded_transformer_paper
    ds_count = ds.count()
    ds.write.duckdb(table_name=table_name, db_url=db_url, dimensions=384)
    conn = duckdb.connect(database=db_url)
    duckdb_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    # delete the database
    try:
        os.unlink(db_url)
    except Exception as e:
        print(f"Error deleting {db_url}: {e}")
    assert ds_count == int(duckdb_count)
