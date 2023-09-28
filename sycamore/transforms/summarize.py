from abc import ABC, abstractmethod
from typing import Callable, Optional

from ray.data import Dataset

from sycamore.data import Element, Document
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Transform, Node
from sycamore.llms import LLM
from sycamore.transforms.map import generate_map_function
from sycamore.llms.prompts import (
    TEXT_SUMMARIZER_GUIDANCE_PROMPT_CHAT,
    TEXT_SUMMARIZER_GUIDANCE_PROMPT,
)


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, document: Document) -> Document:
        pass


class LLMElementTextSummarizer(Summarizer):
    def __init__(self, llm: LLM, element_operator: Optional[Callable[[Document], list[Element]]] = None):
        self._llm = llm
        self._element_operator = element_operator

    def summarize(self, document: Document) -> Document:
        if self._element_operator is not None:
            filtered_elements = self._element_operator(document)
        else:
            filtered_elements = document.elements

        if len(filtered_elements) > 0:
            self._summarize_text_element(filtered_elements)
        return document

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
        dataset = input_dataset.map(generate_map_function(self._summarizer.summarize))
        return dataset
