from typing import Optional, Any, Union

from sycamore.data import Element, Document
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Node
from sycamore.llms import LLM
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
from jinja2.sandbox import SandboxedEnvironment


class LLMTextQueryAgent:
    """
    LLMTextQueryAgent uses a specified LLM to execute LLM queries about a document

    Args:
        prompt: A prompt to be passed into the underlying LLM execution engine
        llm: (LLM) An instance of the LLM class to be pass into the user
        output_property: (Optional) The output property to add results in. Defaults to 'llm_response'
        format_kwargs: (Optional) Formatting arguments passed in to define the prompt, uses a Jinja Sandbox
        number_of_elements: (Optional) Parameter to either limit the number of elements or to add an llm response to the
        entire document using a prefix of elements
        llm_kwargs: (Optional) LLM keyword argument for the underlying execution engine
        per_element: (Optional) Whether to execute the call per each element or on the Document itself. Defaults to
        True.

    Example:
         .. code-block:: python

            prompt="Tell me the important numbers from this element"
            llm_query_agent = LLMElementTextSummarizer(prompt=prompt)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner())
                .llm_query(query_agent=llm_query_agent)
    """

    def __init__(
        self,
        prompt: str,
        llm: LLM,
        output_property: str = "llm_response",
        format_kwargs: Optional[dict[str, Any]] = None,
        number_of_elements: Optional[int] = None,
        llm_kwargs: dict = {},
        per_element: bool = True,
    ):
        self._llm = llm
        self._prompt = prompt
        self._output_property = output_property
        self._llm_kwargs = llm_kwargs
        self._per_element = per_element
        self._format_kwargs = format_kwargs
        self._number_of_elements = number_of_elements

    def execute_query(self, document: Document) -> Document:
        if self._per_element:
            elements = document.elements
            for idx, element in enumerate(document.elements):
                elements[idx] = self._query_text_object(element)
                if self._number_of_elements and idx >= self._number_of_elements:
                    break
            document.elements = elements
        elif self._number_of_elements:  # limit to a number of elements
            text_representation = self._prompt
            for idx, element in enumerate(document.elements):
                text_representation += "\n" + element["text_representation"]
                if idx >= self._number_of_elements:
                    break
            prompt_kwargs = {"prompt": text_representation}
            llm_resp = self._llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs=self._llm_kwargs)
            document["properties"][self._output_property] = llm_resp
        else:
            if document.text_representation:
                document = self._query_text_object(document)
        return document

    @timetrace("LLMQueryText")
    def _query_text_object(self, object: Union[Document, Element]) -> Union[Document, Element]:
        if object.text_representation:
            if self._format_kwargs:
                prompt = (
                    SandboxedEnvironment()
                    .from_string(source=self._prompt, globals=self._format_kwargs)
                    .render(doc=object)
                )
            else:
                prompt = self._prompt + "\n" + object["text_representation"]
            prompt_kwargs = {"prompt": prompt}
            llm_resp = self._llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs=self._llm_kwargs)
            object["properties"][self._output_property] = llm_resp
        return object


class LLMQuery(NonCPUUser, NonGPUUser, Map):
    """
    The LLM Query Transform executes user defined queries on a document or the elements within it.
    """

    def __init__(self, child: Node, query_agent: LLMTextQueryAgent, **kwargs):
        super().__init__(child, f=query_agent.execute_query, **kwargs)
