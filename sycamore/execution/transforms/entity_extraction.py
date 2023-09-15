from abc import ABC, abstractmethod
from typing import Callable, Any, Optional

from ray.data import Dataset

from sycamore.data import Element, Document
from sycamore.execution import Node, Transform
from sycamore.execution.transforms.llms import LLM
from sycamore.execution.transforms.prompts.default_prompts import (
    ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT,
    ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT_CHAT,
    ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT_CHAT,
    ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT,
)


def element_list_formatter(elements: list[Element]) -> str:
    query = ""
    for i in range(len(elements)):
        query += f"ELEMENT {i + 1}: {elements[i]['content']['text']}\n"
    return query


class EntityExtractor(ABC):
    def __init__(self, entity_name: str):
        self._entity_name = entity_name

    @abstractmethod
    def extract_entity(self, record: dict[str, Any]) -> dict[str, Any]:
        pass


class OpenAIEntityExtractor(EntityExtractor):
    def __init__(
        self,
        entity_name: str,
        llm: LLM,
        prompt_template: Optional[str] = None,
        num_of_elements: int = 10,
        prompt_formatter: Callable[[list[Element]], str] = element_list_formatter,
    ):
        super().__init__(entity_name)
        self._llm = llm
        self._num_of_elements = num_of_elements
        self._prompt_template = prompt_template
        self._prompt_formatter = prompt_formatter

    def extract_entity(self, record: dict[str, Any]) -> dict[str, Any]:
        document = Document(record)

        if self._prompt_template:
            entities = self._handle_few_shot_prompting(document)
        else:
            entities = self._handle_zero_shot_prompting(document)

        document.properties.update({f"{self._entity_name}": entities["answer"]})
        return document.to_dict()

    def _handle_few_shot_prompting(self, document: Document) -> Any:
        sub_elements = [document.elements[i] for i in range((min(self._num_of_elements, len(document.elements))))]

        if self._llm.is_chat_mode:
            prompt = ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT_CHAT

        else:
            prompt = ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT

        entities = self._llm.generate(
            prompt_kwargs={
                "prompt": prompt,
                "entity": self._entity_name,
                "examples": self._prompt_template,
                "query": self._prompt_formatter(sub_elements),
            }
        )
        return entities

    def _handle_zero_shot_prompting(self, document: Document) -> Any:
        sub_elements = [document.elements[i] for i in range((min(self._num_of_elements, len(document.elements))))]

        if self._llm.is_chat_mode:
            prompt = ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT_CHAT

        else:
            prompt = ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT

        entities = self._llm.generate(
            prompt_kwargs={"prompt": prompt, "entity": self._entity_name, "query": self._prompt_formatter(sub_elements)}
        )
        return entities


class ExtractEntity(Transform):
    def __init__(
        self,
        child: Node,
        entity_extractor: EntityExtractor,
        **resource_args,
    ):
        super().__init__(child, **resource_args)
        self._entity_extractor = entity_extractor

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self._entity_extractor.extract_entity)
        return dataset
