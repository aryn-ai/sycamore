from typing import Any, Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class Filter(LogicalOperator):
    """
    Logical filter operator for exact match or range filtering.
    """

    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None):
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        *Filter**: Basic filters for data when field already exists. Used in cases where LLM filter
            is not necessary.
        Parameters are *description*, *rangeFilter*, *query*, *start*, *end*, *field*, *input*,
            and *id*. Returns a database.
        - *description* is a written description of the purpose of this operation in this context
            and justification of why you chose to use it.
        - *rangeFilter* is a Boolean. If true, it will use an inclusive range filter (in which case
            you need to specify *start* and/or *end*). The range filter requires a specific field
            value to fall within a range of values. This is mainly used for date ranges,
            e.g. *start*=2022/10/01, *end*=2022/10/20, *field*=date requires the Date to be between
            October 1 and October 20 (inclusive) in 2022. For range filters, you are not required
            to specify both *start* and *end* if unnecessary. If false, it will use a match filter
            for matches (in which case you need to specify *query*). The match filter requires a
            specific field to match a fixed value (the *query*), e.g. match "California" in
            "location".
        - *query* is the value to search for when using a match filter.
        - *start* is the start value for the range filter.
        - *end* is the end value for the range filter.
        - *field* is the name of the database field to filter based on.
        - *date* is a Boolean that specifies if a range filter is being performed on a date.
        - *input* is a list of operation ids that this operation depends on. For this operation,
            *input* should only contain one id of an operation that returns a database
            (len(input) == 1).
        - *id* is a uniquely assigned integer that serves as an identifier.
        """

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = {
            "operatorName": '"Filter"',
            "description": "string",
            "rangeFilter": "Boolean",
            "query": "string",
            "start": "string",
            "end": "string",
            "field": "string",
            "date": "Boolean",
            "input": "number array",
            "id": "number",
        }
        return schema
