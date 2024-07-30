from sycamore.query.operators.logical_operator import LogicalOperator


class LoadData(LogicalOperator):
    """Loads data from a specified index.

    Parameters are *description*, *index*, *query*, and *id*. Returns a database with fields 
    from the schema.

    - *description* is a written description of the purpose of this operation in this context
        and justification of why you chose to use it.
    - *index* is the index to load data from.
    - *query* is the initial query to search for when loading data (so that only the relevant
        data is used).
    - *id* is a uniquely assigned integer that serves as an identifier.
    """

    index: str
    query: str