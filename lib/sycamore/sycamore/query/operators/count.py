from typing import Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class Count(LogicalOperator):
    """Determines the length of a particular database (number of records).
    
    Parameters are *description*, *input*, *field*, *primaryField*, and *id*.
    
    Returns a number.

    - *description* is a written description of the purpose of this operation in this context
        and justification of why you chose to use it.
    - *input* is a list of operation ids that this operation depends on. For this operation,
        *input* should only contain one id of an operation that returns a database (len(input) == 1).
    - *field* is an optional parameter. non-primary database field to return a count based on.
    - *primaryField* is an optional parameter. It is a primary field that represents what a
        unique entry is considered for the data provided.
    - *node_id* is a uniquely assigned integer that serves as an identifier.
    """

    field: Optional[str] = None
    primaryField: Optional[str] = None