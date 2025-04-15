from typing import Optional

from sycamore.query.logical_plan import Node


class KMeanClustering(Node):
    """Group documents based on a particular field.

    Returns a database with ONLY 2 FIELDS: "properties.key" (which corresponds to unique values of
    *field*) and "properties.count" (which contains the counts corresponding to unique values
    of *field*).
    """

    field: Optional[str] = None
    """The database field to find the top K occurences for."""

    new_field: str = "centroids"
    """The centroid field used for clustering"""

    K: int = 5
    """The number of groups."""
