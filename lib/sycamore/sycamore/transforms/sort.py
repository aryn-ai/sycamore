from typing import Any, Optional, TYPE_CHECKING

from sycamore.plan_nodes import Node, Transform
from sycamore.data import Document

if TYPE_CHECKING:
    from ray.data import Dataset


class Sort(Transform):
    """
    Sort by field in Document
    """

    def __init__(self, child: Node, descending: bool, field: str, default_val: Optional[Any] = None):
        super().__init__(child)
        self._descending = descending
        self._field = field
        self._default_val = default_val

    def execute(self, **kwargs) -> "Dataset":
        # creates dataset
        ds = self.child().execute(**kwargs)

        # adds a "key" column containing desired field
        map_fn = self.make_map_fn_sort()
        ds = ds.map(map_fn)

        # sorts the dataset
        ds = ds.sort("key", descending=self._descending)
        ds = ds.drop_columns(["key"])
        return ds

    def make_map_fn_sort(self):
        def ray_callable(input_dict: dict[str, Any]) -> dict[str, Any]:
            doc = Document.from_row(input_dict)

            val = doc.field_to_value(self._field)

            if val is None:
                if self._default_val is None:
                    exception_string = f'Field "{self._field}" not present in Document and default value not provided.'
                    raise Exception(exception_string)
                else:
                    val = self._default_val

            # updates row to include new col
            new_doc = doc.to_row()
            new_doc["key"] = val

            return new_doc

        return ray_callable
