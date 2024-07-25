from typing import Any, Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class LlmExtract(LogicalOperator):
    def __init__(self, node_id: str, data: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        *LlmExtract**: Adds a new field to the inputted database based on extracting information
            from an existing field. 
        Parameters are *description*, *question*, *field*, *newField*, *format*, *input*, and *id*.
        Returns a database.
        - *description* is a written description of the purpose of this operation in this context
            and justification of why you chose to use it.
        - *question* is the prompt to the LLM for creating the new field. Be descriptive with the 
            question and include examples if possible.
        - *field* is the name of the existing field for the LLM to use.
        - *newField* is the name of the new field to add.
        - *format* is the format the new field should be in, e.g. number.
        - *discrete* is a Boolean. It is true ONLY if *newField* has a known finite number of
            possible values (e.g. number, letter, continent, color). It is ALWAYS false otherwise
            (any free text outputs).
        - *input* is a list of operation ids that this operation depends on. For this operation,
            *input* should only contain one id of an operation that returns a database
            (len(input) == 1).
        - *id* is a uniquely assigned integer that serves as an identifier.
        """

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = {
            "operatorName": '"LlmExtract"',
            "description": "string",
            "question": "string",
            "field": "string",
            "newField": "string",
            "format": "string",
            "discrete": "Boolean",
            "input": "number array",
            "id": "number",
        }
        return schema
