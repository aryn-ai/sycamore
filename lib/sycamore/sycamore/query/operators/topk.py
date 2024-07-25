from typing import Any, Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class TopK(LogicalOperator):
    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        **TopK**: Finds the top K frequent occurences of values for a particular field. Parameters 
        are *description*, *field*, *primaryField*, *K*, *descending*, *input*, and *id*. Returns 
        a database with ONLY 2 FIELDS: "properties.key" (which corresponds to unique values of 
        *field*) and "properties.count" (which contains the counts corresponding to unique values 
        of *field*).
        - *description* is a written description of the purpose of this operation in this context
            and justification of why you chose to use it.
        - *field* is the database field to find the top K occurences for.
        - *primaryField* is the an optional parameter. It is a a database field that is required to
            be unique when counting the top K occurences of *field*.
        - *K* is the number of top frequency occurences to look for (e.g. top 2 most common, K=2). 
            If K is null, all occurrences will be returned.
        - *descending* is a boolean. If True, will return the top K most common occurrences.
            If False, will return the top K least common occurrences.
        - *useLLM* is a boolean. If True (SHOULD BE TRUE if *field* is a is a string field in the
            database with INFINITE options), an LLM will be used to identify top K occurrences.
            If False (SHOULD BE FALSE if *field* is a string field with finite options, or is not
            a string), simple database operations will be used.
        - *input* is a list of operation ids that this operation depends on. For this operation,
            *input* should only contain one id of an operation that returns a database
            (len(input) == 1).
        - *id* is a uniquely assigned integer that serves as an identifier.
        """

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = {
            "operatorName": '"TopK"',
            "description": "string",
            "field": "string",
            "primaryField": "string",
            "K": "number",
            "descending": "Boolean",
            "useLLM": "Boolean",
            "input": "number array",
            "id": "number",
        }
        return schema
