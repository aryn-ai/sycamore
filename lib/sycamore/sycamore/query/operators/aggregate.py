from typing import Optional

from pydantic import Field

from sycamore.query.logical_plan import Node


class AggregateCount(Node):
    """Group documents based on a particular field.

    Returns a database with ONLY 2 FIELDS: "properties.key" (which corresponds to unique values of
    *field*) and "properties.count" (which contains the counts corresponding to unique values
    of *field*).
    """

    llm_summary: bool = False
    """If true, use llm_summary to summarize each group; If False, randomly select one entity from each group."""

    llm_summary_instruction: Optional[str] = Field(default=None, json_schema_extra={"exclude_from_comparison": True})
    """An instruction of what the group name should be about if llm_summary is True. E.g. if the
    purpose of this operation is to group by cities, llm_cluster_instruction
    could be 'The city name of this group'"""
