from typing import Optional

from sycamore.data.document import Document
from sycamore.plan_nodes import Node, Write
from sycamore.transforms.map import MapBatch
import glob
import duckdb
import os


class DuckDB_Writer(MapBatch, Write):

    def __init__(
        self,
        plan: Node,
        csv_location: str,
        db_url: str,
        table_name: Optional[str] = None,
        csv_directory_location: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(plan, f=self.write_docs, **kwargs)
        self._filter = filter
        self.csv_location = csv_location
        self.csv_directory_location = csv_directory_location
        self.db_url = db_url
        self.table_name = table_name

    def write_docs(self, docs: list[Document]) -> list[Document]:
        # Check if files are written (to gracefully handle tests to only check execution)
        sql_location = os.path.join(self.csv_location, "*.csv")
        self.table_name = self.table_name if not None else "data"
        if bool(glob.glob(sql_location)):
            client = duckdb.connect(self.db_url)
            client.sql(f"CREATE TABLE {self.table_name} AS SELECT * FROM read_csv('{sql_location}')")
            # Flush out the csv files if not persisted
            if not self.csv_directory_location:
                try:
                    for root, _, files in os.walk(self.csv_location):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.unlink(file_path)
                            except Exception as e:
                                print(f"Error deleting {file_path}: {e}")
                except Exception as e:
                    print(f"Error deleting files in {self.csv_location}: {e}")
        else:
            print(f"No files in directory matching the pattern in {sql_location}")
        return docs
