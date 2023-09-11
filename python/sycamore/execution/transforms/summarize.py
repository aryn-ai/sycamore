from typing import Callable, List, Dict, Any

from ray.data import Dataset

from sycamore.data import Element, Document
from sycamore.execution import NonCPUUser, NonGPUUser, Transform, Node
from sycamore.execution.transforms.llms import LLM


class TextSummarizer:
    def __init__(
        self,
        llm: LLM,
        element_operator: Callable[[Document], List[Element]] = None,
    ):
        self._llm = llm
        self.element_operator = element_operator

    def _summarize_text_element(self, elements: list[Element]):
        if self._llm.is_chat_mode():
            prompt = """
                       {{#system~}}
                       You are a helpful text summarizer.
                       {{~/system}}

                       {{#user~}}
                       Write a summary of the following. Use only the
                       information provided. Include as many key details as
                       possible. Do not make up answer.
                       {{query}}
                       {{~/user}}

                       {{#assistant~}}
                       {{gen "summary"}}
                       {{~/assistant}}
                       """

        else:
            prompt = """ Write a summary of the following. Use only the
            information provided. Include as many key details as possible.
            Do not make up answer.
                           {{query}}
                           =========
                           {{gen "summary"}}
                           """  # noqa: E501

        for element in elements:
            response = self._llm.generate(prompt_kwargs={"prompt": prompt, "query": element.text_representation})
            element.properties["summary"] = response["summary"]

    def summarize(self, row: Dict[str, Any]) -> Dict[str, Any]:
        document = Document(row)
        elements = document.elements

        if self.element_operator is not None:
            elements = self.element_operator(document)

        self._summarize_text_element(elements)
        return document.to_dict()


class SummarizeText(NonCPUUser, NonGPUUser, Transform):
    def __init__(self, child: Node, llm: LLM, element_operator: Callable[[Document], List[Element]] = None, **kwargs):
        super().__init__(child, **kwargs)
        self._text_summarizer = TextSummarizer(llm, element_operator)

    def execute(self) -> Dataset:
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self._text_summarizer.summarize)
        return dataset
