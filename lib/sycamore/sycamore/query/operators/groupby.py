from sycamore.query.logical_plan import Node


class GroupBy(Node):
    """Group documents based on a particular field.

    Returns a database with ONLY 2 FIELDS: "properties.key" (which corresponds to unique values of
    *field*) and "properties.count" (which contains the counts corresponding to unique values
    of *field*).
    """

    field: str = "properties._autogen_ClusterAssignment"
    """The centroid field used for clustering"""


class AggregateCount(Node):
    """Group documents based on a particular field.

    Returns a database with ONLY 2 FIELDS: "properties.key" (which corresponds to unique values of
    *field*) and "properties.count" (which contains the counts corresponding to unique values
    of *field*).
    """


class AggregateCollect(Node):
    """Group documents based on a particular field.

    Returns a database with ONLY 2 FIELDS: "properties.key" (which corresponds to unique values of
    *field*) and "properties.count" (which contains the counts corresponding to unique values
    of *field*).
    """
