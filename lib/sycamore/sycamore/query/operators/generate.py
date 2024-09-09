from sycamore.query.operators.logical_operator import LogicalOperator


class GenerateEnglishResponse(LogicalOperator):
    """
    This operation generates an English response to a user query based on the input data provided.
    This operator should only be used if there is not another, more specific, operator that will
    do a better job of generating the response.

    The response should be in Markdown format. It can contain links, tables, and other
    Markdown elements.

    Whenever possible, provide links to relevant data sources and documents.
    """

    question: str
    """The question to ask the LLM."""


class GenerateTable(LogicalOperator):
    """
    This operation generates a table based on the input data provided.

    The response should be in JSON format. It should contain a list of dictionaries,
    where each dictionary represents a row in the table. For example:

    [
      { "column1": "value1", "column2": "value2" },
      { "column1": "value3", "column2": "value4" },
    ]
    """

    table_definition: str
    """An English description of the table to generate."""


class GeneratePreview(LogicalOperator):
    """
    This operation generates a JSON object showing details or a preview of one or more
    documents provided as input. This is useful if a specific document or small set of documents
    are being requested.

    The response should be in JSON format. It should contain a list of dictionaries,
    where each dictionary represents a document to preview. Each preview should
    contain the fields "path", "title", and "description". For example:
    [
      { "path": "s3://aryn-public/samples/sampledata1.pdf", "title": "Sample Document", "description": "This is a sample document." },
      { "path": "s3://aryn-public/samples/sampledata2.pdf", "title": "Another sample Document", "description": "This is another sample document." },
    ]
    """
