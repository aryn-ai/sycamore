from sycamore.query.operators.logical_operator import LogicalOperator


class LlmFilter(LogicalOperator):
    """LlmFilter uses a Large Language Model (LLM) to filter database records based on the
    value of a field. This operation is useful when the filtering to be performed is complex
    and cannot be expressed either as a OpenSearch query (for QueryDatabase), vector query (QueryVectorDatabase)
    or as a simple range or match filter (for BasicFilter).

    For example, LlmFilter is useful when the filter is operating on the semantic content
    of a field. For example, if a field contains a description of an event and the filter
    is to only include events that are natural disasters, LlmFilter can be used to analyze
    the text of the field to determine if the event is a natural disaster.

    Whenever possible, use the `query' parameter to the QueryDatabase operation or 'query_phrase' parameter
    to the QueryVectorDatabase operation to filter data at the source.

    The BasicFilter operation is preferred to LLMFilter when the filter is simple and does not
    require the complexity of an LLM model to analyze the data.

    Returns a database.
    """

    field: str
    """The name of the field to filter based on."""

    question: str
    """The predicate to filter on. This is a yes/no question in natural language that the LLM will
    use to filter the data. The question should be phrased in a way that the LLM can understand
    and answer. For example, "Is this event a natural disaster?" or "Did this event occur
    outside the United States?"."""

    _keys_to_exclude_for_comparison = {"question"}
