from sycamore.query.operators.logical_operator import LogicalOperator


class LlmGenerate(LogicalOperator):
    """
    LLM generates a conversational English response given a question and its answer
    (or data from which the answer can be determined).

    Parameters are *description*, *question*, *input*, and *id*.

    Returns a string that contains the conversational English response.

    - *description* is a written description of the purpose of this operation in this context 
        and justification of why you chose to use it. Include a thorough description of
        what each of the inputs in *input* is. e.g. input 1 is a database regarding shipwrecks 
        in America and input 2 is a database regarding shipwrecks in Mexico.
    - *question* is the question.
    - *input* is a list of operation ids that this operation depends on. For this operation, 
        *input* should all the ids of operations that help answer the original question.
    - *id* is a uniquely assigned integer that serves as an identifier.
    """

    question: str