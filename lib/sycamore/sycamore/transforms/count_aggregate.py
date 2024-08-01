from typing import Any, Callable, Optional
from ray.data import Dataset
from sycamore.data.document import Document
from sycamore.plan_nodes import Node, Transform


class CountAggregate(Transform):
    """
    Count aggregation that allows you to aggregate by document field(s).
    """

    def __init__(self, child: Node, field: str, unique_field: Optional[str] = None):
        super().__init__(child)
        self._field = field
        self._unique_field = unique_field

    def execute(self, **kwargs) -> "Dataset":
        # creates dataset
        ds = self.child().execute(**kwargs)

        # adds a "key" column containing desired field
        map_fn = self.make_map_fn_count(self._field, self._unique_field)
        ds = ds.map(map_fn)
        ds = ds.filter(lambda row: self.filterOutNone(row))
        # lazy grouping + count aggregation

        if self._unique_field is not None:
            ds = ds.groupby(["key", "unique"]).count()

        ds = ds.groupby("key").count()

        # Add the new column to the dataset
        ds = ds.map(self.add_doc_column)

        return ds

    def make_map_fn_count(
        self, field: str, unique_field: Optional[str] = None
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """
        Creates a map function that can be called on a Ray Dataset
        based on a DocSet. Adds a column to the Dataset based on
        field and unique_field in DocSet documents.

        Args:
            field: Document field to add as a column.
            unique_field: Unique document field to as a column.

        Returns:
            Function that can be called inside of DocSet.filter
        """

        def ray_callable(input_dict: dict[str, Any]) -> dict[str, Any]:
            doc = Document.from_row(input_dict)
            key_val = doc.field_to_value(field)

            if key_val is None:
                return (
                    {"doc": None, "key": None, "unique": None}
                    if unique_field is not None
                    else {"doc": None, "key": None}
                )

            new_doc = doc.to_row()
            new_doc["key"] = key_val

            if unique_field is not None:
                unique_val = doc.field_to_value(unique_field)
                if unique_val is None:
                    return {"doc": None, "key": None, "unique": None}
                new_doc["unique"] = unique_val

            return new_doc

        return ray_callable

    def add_doc_column(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Adds a doc column with serialized document to Ray Dataset.

        Args:
            row: Input Dataset row.

        Returns:
            Row with added doc column.
        """
        row["doc"] = Document(
            text_representation="", properties={"key": row["key"], "count": row["count()"]}
        ).serialize()
        return row

    def filterOutNone(self, row: dict[str, Any]) -> bool:
        """
        Filters out Dataset rows where all values are None.

        Args:
            row: Input Dataset row.

        Returns:
            Boolean that indicates whether or not to keep row.
        """
        return_value = row["doc"] is not None and row["key"] is not None

        if "unique" in row:
            return_value = return_value and row["unique"] is not None

        return return_value
