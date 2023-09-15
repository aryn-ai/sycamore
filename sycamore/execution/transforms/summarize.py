from abc import ABC, abstractmethod
from typing import Callable, Any

from ray.data import Dataset

from sycamore.data import Element, Document
from sycamore.execution import NonCPUUser, NonGPUUser, Transform, Node
from sycamore.execution.transforms.llms import LLM
from sycamore.execution.transforms.prompts.default_prompts import (
    TEXT_SUMMARIZER_GUIDANCE_PROMPT_CHAT,
    TEXT_SUMMARIZER_GUIDANCE_PROMPT,
)


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, record: dict[str, Any]) -> dict[str, Any]:
        pass


class LLMElementTextSummarizer(Summarizer):
    def __init__(self, llm: LLM, element_operator: Callable[[Document], list[Element]] | None = None):
        self._llm = llm
        self._element_operator = element_operator

    def summarize(self, row: dict[str, Any]) -> dict[str, Any]:
        document = Document(row)
        if self._element_operator is not None:
            filtered_elements = self._element_operator(document)
        else:
            filtered_elements = document.elements

        if len(filtered_elements) > 0:
            self._summarize_text_element(filtered_elements)
        return document.to_dict()

    def _summarize_text_element(self, elements: list[Element]):
        if self._llm.is_chat_mode:
            prompt = TEXT_SUMMARIZER_GUIDANCE_PROMPT_CHAT

        else:
            prompt = TEXT_SUMMARIZER_GUIDANCE_PROMPT

        for element in elements:
            if element.text_representation:
                response = self._llm.generate(prompt_kwargs={"prompt": prompt, "query": element.text_representation})
                element.properties["summary"] = response["summary"]


class Summarize(NonCPUUser, NonGPUUser, Transform):
    def __init__(self, child: Node, summarizer: Summarizer, **kwargs):
        super().__init__(child, **kwargs)
        self._summarizer = summarizer

    def execute(self) -> Dataset:
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self._summarizer.summarize)
        return dataset
