from typing import Dict, Any

from sycamore.query.operators.logical_operator import LogicalOperator


class LlmGenerate(LogicalOperator):
    def __init__(self, node_id: str, data: Dict[Any, Any] = None) -> None:
        super().__init__(node_id, data)

    @staticmethod
    def description() -> str:
        return """
        **LlmGenerate**: LLM generates a conversational English response given a question and its answer (or data from which the answer can be determined). 
        Parameters are *description*, *question*, *input*, and *id*. Returns a string that contains the conversational English response.
        - *description* is a written description of the purpose of this operation in this context and justification of why you chose to use it.
        - *question* is the question
        - *input* is a list of operation ids that this operation depends on. for this operation, *input* should only contain the id of an operation that contains the answer to the original question.
        - *id* is a uniquely assigned integer that serves as an identifier
        """

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        schema = {
            "operatorName": '"LlmGenerate"',
            "description": "string",
            "question": "string",
            "input": "number array",
            "id": "number",
        }
        return schema
