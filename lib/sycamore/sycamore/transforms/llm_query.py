from typing import Optional, Any, Union

from sycamore.data import Element, Document
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Node
from sycamore.llms import LLM
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
from jinja2.sandbox import SandboxedEnvironment


class LLMTextQueryAgent:
    """
    LLMTextQueryAgent uses a specified LLM to execute LLM queries about a document or its child elements.

    Args:
        prompt: A prompt to be passed into the underlying LLM execution engine
        llm: The LLM Client to be used here. It is defined as an instance of the LLM class in Sycamore.
        output_property: (Optional, default="llm_response") The output property of the document or element to add
            results in.
        format_kwargs: (Optional, default="None") If passed in, details the formatting details that must be
            passed into the underlying Jinja Sandbox.
        number_of_elements: (Optional, default="None") When "per_element" is true, limits the number of
            elements to add an "output_property". Otherwise, the response is added to the
            entire document using a limited prefix subset of the elements.
        llm_kwargs: (Optional) LLM keyword argument for the underlying execution engine
        per_element: (Optional, default="{}") Keyword arguments to be passed into the underlying LLM execution engine.
        element_type: (Optional) Parameter to only execute the LLM query on a particular element type. If not specified,
            the query will be executed on all elements.
    Example:
         .. code-block:: python

            prompt="Tell me the important numbers from this element"
            llm_query_agent = LLMTextQueryAgent(prompt=prompt)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=ArynPartitioner())
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
        element_type: Optional[str] = None,
        table_cont: Optional[bool] = False,
    ):
        self._llm = llm
        self._prompt = prompt
        self._output_property = output_property
        self._llm_kwargs = llm_kwargs
        self._per_element = per_element
        self._format_kwargs = format_kwargs
        self._number_of_elements = number_of_elements
        self._element_type = element_type
        self._table_cont = table_cont

    def execute_query(self, document: Document) -> Document:
        final_prompt = self._prompt
        element_count = 0
        prev_table = -1
        if self._per_element or self._number_of_elements:
            for idx, element in enumerate(document.elements):
                if self._element_type and element.type != self._element_type:
                    continue
                if self._per_element:
                    if not self._table_cont:
                        document.elements[idx] = self._query_text_object(element)
                    else:
                        if prev_table >= 0:
                            document.elements[idx] = self._query_text_object(element, document.elements[prev_table])
                        else:
                            document.elements[idx] = self._query_text_object(element)
                        prev_table = idx
                else:
                    final_prompt += "\n" + element["text_representation"]
                if self._number_of_elements:
                    element_count += 1
                    if element_count >= self._number_of_elements:
                        break
            if not self._per_element:
                prompt_kwargs = {"prompt": final_prompt}
                llm_resp = self._llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs=self._llm_kwargs)
                document["properties"][self._output_property] = llm_resp
        else:
            if document.text_representation:
                document = self._query_text_object(document)
        return document

    @timetrace("LLMQueryText")
    def _query_text_object(
        self, object: Union[Document, Element], objectPrev: Optional[Element] = None
    ) -> Union[Document, Element]:
        if object.text_representation:
            if self._format_kwargs:
                prompt = (
                    SandboxedEnvironment()
                    .from_string(source=self._prompt, globals=self._format_kwargs)
                    .render(doc=object)
                )
            else:
                object_name = "ELEMENT" if isinstance(object, Element) else "DOCUMENT"
                if objectPrev and objectPrev.text_representation:
                    prompt = (
                        self._prompt
                        + "\n"
                        + f"{object_name} 1: \n\n"
                        + objectPrev.text_representation
                        + "\n\n"
                        + f"{object_name} 2: \n"
                        + object.text_representation
                    )
                else:
                    prompt = self._prompt + "\n" + object.text_representation
            prompt_kwargs = {"prompt": prompt}
            llm_resp = self._llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs=self._llm_kwargs)
            if self._table_cont:
                object["properties"]["table_continuation"] = llm_resp
            else:
                object["properties"][self._output_property] = llm_resp
        return object


class LLMQuery(NonCPUUser, NonGPUUser, Map):
    """
    The LLM Query Transform executes user defined queries on a document or the elements within it.
    """

    def __init__(self, child: Node, query_agent: LLMTextQueryAgent, **kwargs):
        super().__init__(child, f=query_agent.execute_query, **kwargs)
