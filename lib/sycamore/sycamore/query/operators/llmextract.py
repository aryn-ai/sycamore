from sycamore.query.operators.logical_operator import LogicalOperator


class LlmExtract(LogicalOperator):
    """Adds a new field to the input database based on extracting information from an
    existing field.

    Parameters are *description*, *question*, *field*, *newField*, *format*, *input*, and *id*.

    Returns a database.

    - *description* is a written description of the purpose of this operation in this context
        and justification of why you chose to use it.
    - *question* is the prompt to the LLM for creating the new field. Be descriptive with the
        question and include examples if possible.
    - *field* is the name of the existing field for the LLM to use.
    - *new_field* is the name of the new field to add.
    - *new_field_format* is the format the new field should be in, e.g. number.
    - *discrete* is a Boolean. It is true ONLY if *new_field* has a known finite number of
        possible values (e.g. number, letter, continent, color). It is ALWAYS false otherwise
        (any free text outputs).
    - *input* is a list of operation ids that this operation depends on. For this operation,
        *input* should only contain one id of an operation that returns a database
        (len(input) == 1).
    - *id* is a uniquely assigned integer that serves as an identifier.
    """

    question: str
    field: str
    new_field: str
    new_field_format: str
    discrete: bool = False