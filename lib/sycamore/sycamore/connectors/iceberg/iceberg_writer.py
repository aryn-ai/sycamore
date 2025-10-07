from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING 

from sycamore.connectors.base_writer import BaseDBWriter
from sycamore.utils.import_utils import requires_modules

from sycamore.data import Document
from sycamore.plan_nodes import Node, Write
from sycamore.transforms.map import MapBatch
from sycamore.schema import SchemaV2
from sycamore.utils.pyarrow import schema_to_pyarrow, docs_to_pyarrow


if TYPE_CHECKING:
    import pyarrow as pa

    

# This is inspired by the Ray IcebergDataSink class. 
class IcebergWriter(MapBatch):
    def __init__(
            self,
            child: Optional[Node],
            catalog_kwargs: dict[str, Any],
            schema: SchemaV2,
            table_identifier: str,
            **resource_args):
        
        self._catalog_kwargs = catalog_kwargs
        self._schema = schema
        self._pa_schema = schema_to_pyarrow(schema)
        self._table_id = table_identifier
        self._txn = None
        super().__init__(child=child, f=lambda x: x, name="WriteIceberg")

    def _get_catalog(self):
        from pyiceberg.catalog import load_catalog

        catalog = load_catalog(**self._catalog_kwargs)
        return catalog
        
    def _start_txn(self):
        table = self._get_catalog().load_table(self.table_name)
        return table.transaction()

    def write_file_batch(self, docs: list["Document"]) -> list["Document"]:
        pass
        #table = docs_to_pyarrow(docs, self._pa_schema)
        #return docs
    
    def execute(self, **kwargs):
        raise NotImplementedError("Iceberg write does not support distributed execution yet. Use local_execute.")

    def local_execute(self, all_docs: list["Document"]) -> list["Document"]:
        from pyiceberg.exceptions import TableAlreadyExistsError
        from pyiceberg.catalog import Catalog
        
        new_docs = docs_to_pyarrow(all_docs, self._pa_schema)

        ns = Catalog.namespace_from(self._table_id)  # ensure namespace exists
        catalog = self._get_catalog()

        catalog.create_namespace_if_not_exists(ns)

        print(f"In local_execute {catalog.list_namespaces()=}")

        table = catalog.create_table_if_not_exists(self._table_id, self._pa_schema)

        table.append(new_docs)

        print(catalog.list_tables(ns))

        self.table = table
        
# @dataclass
# class IcebergWriterClientParams(BaseDBWriter.ClientParams):
#     catalog_name: str
#     uri: str
#     warehouse: str
#     username: str
#     password: str


# @dataclass
# class IcebergWriterTargetParams(BaseDBWriter.TargetParams):
#     table_name: str
    
    
# class IcebergClient(BaseDBWriter.Client):

#     @requires_modules("pyiceberg", extra="iceberg")
#     def __init__(self, client_params: IcebergWriterClientParams):
#         pass

    
#     def from_client_params(cls, params: "BaseDBWriter.ClientParams") -> "BaseDBWriter.Client":
#         pass

    
#     def write_many_records(self, records: list["BaseDBWriter.Record"], target_params: "BaseDBWriter.TargetParams"):
#         pass


#     def create_target_idempotent(self, target_params: "BaseDBWriter.TargetParams"):
#         pass

    
#     def get_existing_target_params(self, target_params: "BaseDBWriter.TargetParams") -> "BaseDBWriter.TargetParams":
#         pass


# class IcebergRecord(BaseDBWriter.Record):
#     pass
