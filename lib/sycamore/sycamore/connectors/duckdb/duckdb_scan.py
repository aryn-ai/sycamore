from sycamore.plan_nodes import Scan
from sycamore.data import Document
from ray.data import Dataset, from_items
import duckdb


class DuckDBScan(Scan):
    def __init__(self, table_name: str = "default_table", db_url: str = "tmp.db", **kwargs):
        super().__init__(**kwargs)
        self._table_name = table_name
        self._db_url = db_url

    def execute(self) -> Dataset:
        documents = []
        con = duckdb.connect(database=self._db_url, read_only=True)
        data = con.execute(f"SELECT * from {self._table_name}").fetchdf().to_dict(orient="records")
        for object in data:
            doc = Document(object)
            documents.append(doc)

        return from_items(items=[{"doc": doc.serialize()} for doc in documents])

    def format(self):
        return "duckdb"
