from typing import Optional

from pydantic import Field

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

    K: Optional[int] = None
    """The number of groups."""


class LLMClustering(Node):
    """Group documents based on a particular field.

    Returns a database with ONLY 2 FIELDS: "properties.key" (which corresponds to unique values of
    *field*) and "properties.count" (which contains the counts corresponding to unique values
    of *field*).
    """

    field: str
    """The database field to find the top K occurences for."""

    new_field: str = "_autogen_ClusterAssignment"
    """The field for cluster or group assignment"""

    llm_group_instruction: Optional[str] = Field(default=None, json_schema_extra={"exclude_from_comparison": True})
    """An instruction of what the groups should be about E.g. if the
    purpose of this operation is to find the top 2 most frequent cities, llm_cluster_instruction
    could be 'Form groups of different cities'"""
