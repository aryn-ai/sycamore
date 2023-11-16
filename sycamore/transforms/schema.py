from abc import ABC, abstractmethod
from typing import Callable, Any, Optional
import json
import re

from ray.data import Dataset

from sycamore.data import Element, Document
from sycamore.plan_nodes import Node, Transform
from sycamore.llms import LLM
from sycamore.transforms.map import generate_map_function, generate_map_batch_function
from sycamore.llms.prompts import (
    SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT,
    SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT_CHAT,
    PROPERTIES_ZERO_SHOT_GUIDANCE_PROMPT,
    PROPERTIES_ZERO_SHOT_GUIDANCE_PROMPT_CHAT,
)


def element_list_formatter(elements: list[Element]) -> str:
    query = ""
    for i in range(len(elements)):
        query += f"ELEMENT {i + 1}: {elements[i].text_representation}\n"
    return query


class SchemaExtractor(ABC):
    def __init__(self, entity_name: str):
        self._entity_name = entity_name

    @abstractmethod
    def extract_schema(self, document: Document) -> Document:
        pass

class PropertyExtractor(ABC):
    def __init__(self,):# properties: list[str]):
       # self._properties = properties
       pass

    @abstractmethod
    def extract_properties(self, document: Document) -> Document:
        pass


class OpenAISchemaExtractor(SchemaExtractor):
    """
    OpenAISchema uses one of OpenAI's language model (LLM) for schema extraction.
    """

    def __init__(
        self,
        entity_name: str,
        llm: LLM,
        num_of_elements: int = 35,
        prompt_formatter: Callable[[list[Element]], str] = element_list_formatter,
    ):
        super().__init__(entity_name)
        self._llm = llm
        self._num_of_elements = num_of_elements
        self._prompt_formatter = prompt_formatter

    def extract_schema(self, document: Document) -> Document:
        if isinstance(document, list):
            target = document[0]
        else:
            target = document

        entities = self._handle_zero_shot_prompting(target)

        try:
            payload = entities["answer"]
            pattern = r'```json([\s\S]*?)```'
            match = re.match(pattern, payload).group(1)
            answer = json.loads(match)
        except:
            answer = entities["answer"]

        if isinstance(document, list):
            entries = document
        else:
            entries = [ document ]

        for x in entries:
            properties = x.properties
            properties.update({"_schema": answer, "_schema_class": self._entity_name})
            x.properties = properties

        return document

    def _handle_zero_shot_prompting(self, document: Document) -> Any:
        sub_elements = [document.elements[i] for i in range((min(self._num_of_elements, len(document.elements))))]

        if self._llm.is_chat_mode:
            prompt = SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT_CHAT

        else:
            prompt = SCHEMA_ZERO_SHOT_GUIDANCE_PROMPT

        entities = self._llm.generate(
            prompt_kwargs={"prompt": prompt, "entity": self._entity_name, "query": self._prompt_formatter(sub_elements)}
        )

        return entities

class OpenAIPropertyExtractor(PropertyExtractor):
    """
    OpenAISchema uses one of OpenAI's language model (LLM) to extract property values once schema is established.
    """

    def __init__(
        self,
        #properties: list[str],
        llm: LLM,
        num_of_elements: int = 10,
        prompt_formatter: Callable[[list[Element]], str] = element_list_formatter,
    ):
        super().__init__()
        self._llm = llm
        self._num_of_elements = num_of_elements
        self._prompt_formatter = prompt_formatter

    def extract_properties(self, document: Document) -> Document:
        entities = self._handle_zero_shot_prompting(document)

        try:
            payload = entities["answer"]
            pattern = r'```json([\s\S]*?)```'
            match = re.match(pattern, payload).group(1)
            answer = json.loads(match)
        except:
            answer = entities["answer"]

        properties = document.properties
        properties.update({"entity": answer})
        document.properties = properties

        return document

    def _handle_zero_shot_prompting(self, document: Document) -> Any:
        if document.text_representation:
            text = document.text_representation
        else:
            text = [document.elements[i] for i in range((min(self._num_of_elements, len(document.elements))))]

        if self._llm.is_chat_mode:
            prompt = PROPERTIES_ZERO_SHOT_GUIDANCE_PROMPT_CHAT
        else:
            prompt = PROPERTIES_ZERO_SHOT_GUIDANCE_PROMPT

        schema_name = document.properties["_schema_class"]
        schema = document.properties["_schema"]

        entities = self._llm.generate(
            prompt_kwargs={"prompt": prompt, "entity": schema_name, "properties": schema, "query": text}
        )
        return entities

class ExtractSchema(Transform):
    """
    ExtractSchema is a transformation class for extracting a schema from a document using an SchemaExtractor.
    """

    def __init__(
        self,
        child: Node,
        schema_extractor: SchemaExtractor,
        **resource_args,
    ):
        super().__init__(child, **resource_args)
        self._schema_extractor = schema_extractor

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        dataset = input_dataset.map(generate_map_function(self._schema_extractor.extract_schema))
        return dataset

class ExtractBatchSchema(Transform):
    """
    ExtractBatchSchema is a transformation class for extracting a schema from a dataset using an SchemaExtractor.
    This assumes all documents in the dataset share a common schema.
    """

    def __init__(self, child: Node, schema_extractor: SchemaExtractor, **resource_args):
        super().__init__(child, **resource_args)
        self._schema_extractor = schema_extractor

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        n = dataset.count()

        output = dataset.map_batches(
            generate_map_batch_function(self._schema_extractor.extract_schema),
            batch_size=n,
            **self.resource_args
        )

        return output

class ExtractProperties(Transform):
    """
    ExtractEntity is a transformation class for extracting a schema from a dataset using an SchemaExtractor.
    """

    def __init__(
        self,
        child: Node,
        property_extractor: PropertyExtractor,
        **resource_args,
    ):
        super().__init__(child, **resource_args)
        self._property_extractor = property_extractor

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        dataset = input_dataset.map(generate_map_function(self._property_extractor.extract_properties))
        return dataset