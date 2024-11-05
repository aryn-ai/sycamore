from typing import Optional

from sycamore import DocSet
from sycamore.query.logical_plan import Node


class Count(Node):
    """Returns a count of the number of database records provided as input. Optionally supports
    a distinct_field parameter to count the number of distinct values of a given field. For example,
    if distinct_field is 'incident_id', the count will return the number of unique incident_id values
    in the input database records. Otherwise, the count will return the total number of input records.

    Note that you almost always want to use distinct_field, unless you are certain that each
    of the input records represents a unique entity that you wish to count.

    Returns a number.
    """

    distinct_field: Optional[str] = None
    """If specified, returns the count of distinct values of this field in the input.
    If unspecified, returns the count of all input records.
    """

    @property
    def input_types(self) -> set[type]:
        return {DocSet}

    @property
    def output_type(self) -> type:
        return int
