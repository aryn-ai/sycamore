from typing import Any, Optional
from abc import ABC, abstractmethod
from sycamore.data.document import Document
from dateutil import parser


class BasicFilter(ABC):
    def __init__(self, field: str):
        self._field = field

    @abstractmethod
    def __call__(self, document: Document) -> bool:
        pass


class MatchFilter(BasicFilter):
    """
    Only keep documents that match the query on the specified field.
    Performs substring matching for strings.

    Args:
        query: Query to match for.
        field: Document field that is used for filtering.
        ignore_case: Determines case sensitivity for strings.

    Returns:
        A filtered DocSet.
    """

    def __init__(self, field: str, query: Any, ignore_case: bool = True):
        super().__init__(field)
        self._query = query
        self._ignore_case = ignore_case

    def __call__(self, doc: Document) -> bool:
        value = doc.field_to_value(self._field)

        # substring matching
        if isinstance(self._query, str) or isinstance(value, str):
            query_str = str(self._query)
            value_str = str(value)
            if self._ignore_case:
                value_str = value_str.lower()
                query_str = query_str.lower()

            return query_str in value_str

        # For non-string types, check exact match
        return self._query == value


class RangeFilter(BasicFilter):
    """
    Only keep documents for which the value of the
    specified field is within the start:end range.

    Args:
        field: Document field to filter based on.
        start: Value for start of range.
        end: Value for end of range.
        date: Indicates whether start:end is a date range.

    Returns:
        A filtered DocSet.
    """

    def __init__(
        self,
        field: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        date: Optional[bool] = False,
    ):
        super().__init__(field)
        self._start = start
        self._end = end
        self._date = date

    def __call__(self, doc: Document) -> bool:
        value = doc.field_to_value(self._field)

        if self._date:
            if not isinstance(value, str):
                raise ValueError("value must be a string for date filtering")
            value_comp = self.to_date(value)
            if self._start and not isinstance(self._start, str):
                raise ValueError("start must be a string for date filtering")
            start_comp = parser.parse(self._start).replace(tzinfo=None) if self._start else None
            if self._end and not isinstance(self._end, str):
                raise ValueError("end must be a string for date filtering")
            end_comp = parser.parse(self._end).replace(tzinfo=None) if self._end else None
        else:
            value_comp = value
            start_comp = self._start
            end_comp = self._end

        if start_comp is None:
            if end_comp is None:
                raise ValueError("At least one of start or end must be specified")
            return value_comp <= end_comp
        if end_comp is None:
            if start_comp is None:
                raise ValueError("At least one of start or end must be specified")
            return value_comp >= start_comp
        return value_comp >= start_comp and value_comp <= end_comp

    def to_date(self, date_string: str):
        return parser.parse(date_string).replace(tzinfo=None)
