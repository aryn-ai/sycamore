from typing import Any, Callable, Optional, TYPE_CHECKING

from sycamore.utils.import_utils import requires_modules

from sycamore.data import Document, MetadataDocument
from sycamore.plan_nodes import Node, Write
from sycamore.schema import SchemaV2
from sycamore.utils.pyarrow import schema_to_pyarrow, docs_to_pyarrow


if TYPE_CHECKING:
    import pyarrow as pa


class IcebergWriter(Write):

    @requires_modules(["pyiceberg"], extra="iceberg")
    def __init__(
        self,
        child: Node,
        catalog_kwargs: dict[str, Any],
        schema: SchemaV2,
        table_identifier: str,
        property_root: str = "entity",
        location: Optional[str] = None,
        **resource_args,
    ):
        self._catalog_kwargs = catalog_kwargs
        self._schema = schema
        self._pa_schema = schema_to_pyarrow(schema)
        self._table_id = table_identifier
        self._property_root = property_root
        self._location = location
        super().__init__(child=child, f=self, name="WriteIceberg")

    def _get_catalog(self):
        from pyiceberg.catalog import load_catalog

        catalog = load_catalog(**self._catalog_kwargs)
        return catalog

    def __str__(self):
        return f"iceberg_writer(table_id={self._table_id})"

    def _get_table(self):
        from pyiceberg.catalog import Catalog

        ns = Catalog.namespace_from(self._table_id)
        catalog = self._get_catalog()
        catalog.create_namespace_if_not_exists(ns)
        return catalog.create_table_if_not_exists(self._table_id, self._pa_schema, location=self._location)

    def _to_property_dict(self, property_root: str = "entity") -> Callable[["pa.Table"], "pa.Table"]:
        schema = self._pa_schema

        def f(batch: "pa.Table") -> "pa.Table":
            doc_dict = batch.to_pydict()
            all_docs = [Document.deserialize(s) for s in doc_dict["doc"]]
            docs = [d for d in all_docs if not isinstance(d, MetadataDocument)]
            return docs_to_pyarrow(docs, schema)

        return f

    def execute(self, **kwargs):
        _ = self._get_table()  # Creates the table if it does not exist.
        dataset = self.child().execute(**kwargs)
        dataset.map_batches(self._to_property_dict(), batch_format="pyarrow").write_iceberg(
            self._table_id, catalog_kwargs=self._catalog_kwargs, **kwargs
        )
        return dataset

    def local_execute(self, all_docs: list["Document"]) -> list["Document"]:
        table = self._get_table()

        new_docs = docs_to_pyarrow(all_docs, self._pa_schema)
        table.append(new_docs)

        return all_docs
