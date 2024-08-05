from typing import Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class TopK(LogicalOperator):
    """Finds the top K frequent occurences of values for a particular field.

    Returns a database with ONLY 2 FIELDS: "properties.key" (which corresponds to unique values of
    *field*) and "properties.count" (which contains the counts corresponding to unique values
    of *field*).
    """

    field: str
    """The database field to find the top K occurences for."""

    primary_field: Optional[str] = None
    """A database field that is required to be unique when counting the top K occurences of *field*."""

    K: int
    """The number of top frequency occurences to look for (e.g. top 2 most common, K=2)."""

    descending: bool = False
    """If True, will return the top K most common occurrences. If False, will return the top K
    least common occurrences."""

    llm_cluster: bool = False
    """If True (SHOULD BE TRUE if *field* is a is a string field in the database with an unbound
    number of possible values), an LLM will be used to identify top K occurrences. If False
    (SHOULD BE FALSE if *field* is a string field with a bounded number of possible values, or
    is not a string), simple database operations will be used."""

    llm_cluster_description: Optional[str] = None
    """A description of what the groups should be about if llm_cluster is True. E.g. if the
    purpose of this operation is to find the top 2 most frequent cities, llm_cluster_description
    could be 'Form groups of different cities'"""
