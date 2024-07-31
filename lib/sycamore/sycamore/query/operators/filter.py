from typing import Any, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class Filter(LogicalOperator):
    """Basic filters for data when field already exists. Used in cases where LLM filter is not
    necessary.

    Returns a database.
    """

    range_filter: bool = False
    """If true, it will use an inclusive range filter (in which case you need to specify *start*
    and/or *end*). The range filter requires a specific field value to fall within a range of values.
    This is mainly used for date ranges, e.g. *start*=2022/10/01, *end*=2022/10/20, *field*=date
    requires the Date to be between October 1 and October 20 (inclusive) in 2022. For range filters,
    you are not required to specify both *start* and *end* if unnecessary. If false, it will use a
    match filter for matches (in which case you need to specify *query*). The match filter requires a
    specific field to match a fixed value (the *query*), e.g. match "California" in "location".
    """

    query: Optional[Any] = None
    """The value to search for when using a match filter."""

    start: Optional[Any] = None
    """The start value for the range filter."""

    end: Optional[Any] = None
    """The end value for the range filter."""

    field: str
    """The name of the database field to filter based on."""

    date: bool = False
    """Specifies if the range filter is being performed on a date."""
