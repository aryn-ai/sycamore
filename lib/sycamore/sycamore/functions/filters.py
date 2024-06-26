from typing import Optional, Any
from sycamore.llms.openai import OpenAI
from sycamore.data import Document


def llm_filter(
    client: OpenAI,
    doc: Document,
    filter_question: str,
    text: Optional[str] = None,
    filter_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    threshold: int = 3,
) -> bool:
    """This operation filters your DocSet to only keep documents that score greater
    than or equal to the inputted threshold value from an LLM call that returns an int.

    Args:
        client: The Sycamore OpenAI client to use
        doc: The document in the DocSet
        text: The text to filter based on, default is doc.text_representation
        filter_question: Question for filter
        filter_prompt: Custom prompt that you can use for filtering
        system_prompt: Custom prompt to the system
        threshold: Threshold for success, e.g. 3 (default scale is 0-5)

    Returns:
        A boolean that indicates whether or not the text was accepted by the LLM Filter

    Example:
        client = OpenAI(OpenAIModels.GPT_4O.value)
        filter_question = "Was this incident caused by environmental factors?"

        def wrapper(doc: Document) -> bool:
            return llm_filter(client, doc, filter_question)

        docset = (docset.filter(wrapper))
    """
    if text is None:
        text = doc.text_representation

    if system_prompt is None:
        system_prompt = "You are a helpful classifier that generously filters database entries based on questions."

    if filter_prompt is None:
        filter_prompt = f"""Given an entry and a question, you will answer the question relating to the entry. 
                You only respond with 0, 1, 2, 3, 4, or 5 based on your confidence level. 0 is the most negative 
                answer and 5 is the most positive answer. Question: {filter_question}; Entry: {text}"""

    # sets prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": filter_prompt,
        },
    ]
    prompt_kwargs = {"messages": messages}

    # call to LLM
    completion = client.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    # determines if llm output >= threshold
    try:
        return_value = int(completion.content) >= threshold
    except:
        # accounts for llm output errors
        return_value = False
        raise

    return return_value


def match_filter(query: Any, input: Any) -> bool:
    """This operation filters your Docset to only keep documents that match the
    query on the specified input. If the query/input are strings, it looks for
    a substring match. For any type other than strings, it looks for an exact match.

    Args:
        query: The query to filter based on
        input: The input to search for a match in

    Returns:
        A boolean indicating if the input was accepted in the match filter

    Example:
        def wrapper(doc: Document) -> bool:
            query = "Cessna"
            input = doc.properties["entity"]["aircraft"]
            return match_filter(query, input)

        docset = docset.filter(wrapper)
    """

    # substring matching
    if isinstance(query, str) and isinstance(input, str):
        return query in input

    # if not string, exact match
    return query == input


def range_filter(start: Any, end: Any, input: Any) -> bool:
    """This operation filters your Docset to only keep documents for which the value of the
    specified input is within the start:end range.

    Args:
        start: The start value for the range
        end: The end value for the range
        input: The input to run the range filter on

    Returns:
        A boolean indicating if the input was accepted in the range filter

    Example:
        def wrapper(doc: Document) -> bool:
            start, end = 2, 4
            input = doc.properties["page_number"]
            return range_filter(start, end, input)

        docset = docset.filter(wrapper)
    """
    return input >= start and input <= end
