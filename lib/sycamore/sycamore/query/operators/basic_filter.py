from typing import Any, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class BasicFilter(LogicalOperator):
    """Filter data based on a simple range or match filter.

    Whenever possible, use the `query' parameter to the QueryDatabase operation to filter
    data at the source, as this is more efficient than use of the BasicFilter operation.

    The BasicFilter operation is preferred to LLMFilter when the filter is simple and does not
    require the complexity of an LLM model to analyze the data.

    Returns a database.
    """

    range_filter: bool = False
    """If range_filter is true, performs an inclusive range filter (in which case you need to specify *start*
    and/or *end*). The range filter requires a specific field value to fall within a range of values.
    This is mainly used for date ranges. For example, start=2022/10/01, end=2022/10/20, field="date"
    requires the "date" field to be between October 1 and October 20 (inclusive) in 2022.
    At least one of *start* or *end* must be specified. If only one is specified, the filter will be
    open-ended. If both are specified, the filter will be closed-ended.

    If range_filter is false, a match filter will be used (in which case you need to specify *query*).
    The match filter requires a specific field to match a fixed value (the *query*), 
    e.g. match 2 in "passenger_count". For strings, a match filter performs substring matching.
    For example, e.g. query="sub" field="vehicle" would match with the value "submarine" in the
    "vehicle" field.
    """

    query: Optional[Any] = None
    """The value to search for when using a match filter."""

    start: Optional[Any] = None
    """The start value for a range filter."""

    end: Optional[Any] = None
    """The end value for a range filter."""

    field: str
    """The name of the database field to filter based on."""

    date: bool = False
    """Specifies if the range filter is being performed on a date."""
