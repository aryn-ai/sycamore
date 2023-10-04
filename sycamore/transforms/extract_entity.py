from abc import ABC, abstractmethod
from typing import Callable, Any, Optional

from ray.data import Dataset

from sycamore.data import Element, Document
from sycamore.plan_nodes import Node, Transform
from sycamore.llms import LLM
from sycamore.transforms.map import generate_map_function
from sycamore.llms.prompts import (
    ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT,
    ENTITY_EXTRACTOR_ZERO_SHOT_GUIDANCE_PROMPT_CHAT,
    ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT_CHAT,
    ENTITY_EXTRACTOR_FEW_SHOT_GUIDANCE_PROMPT,
)


def element_list_formatter(elements: list[Element]) -> str:
    query = ""
    for i in range(len(elements)):
        query += f"ELEMENT {i + 1}: {elements[i]['text_representation']}\n"
    return query


class EntityExtractor(ABC):
    def __init__(self, entity_name: str):
        self._entity_name = entity_name

    @abstractmethod
    def extract_entity(self, document: Document) -> Document:
        pass


class OpenAIEntityExtractor(EntityExtractor):
    """
    OpenAIEntityExtractor uses one of OpenAI's language model (LLM) for entity extraction.

    This class inherits from EntityExtractor and is designed for extracting a specific entity from a document using
    OpenAI's language model. It can use either zero-shot prompting or few-shot prompting to extract the entity.
    The extracted entities from the input document are put into the document properties.

    Args:
        entity_name: The name of the entity to be extracted.
        llm: An instance of an OpenAI language model for text processing.
        prompt_template: A template for constructing prompts for few-shot prompting. Default is None.
        num_of_elements: The number of elements to consider for entity extraction. Default is 10.
        prompt_formatter: A callable function to format prompts based on document elements.

    Example:
        .. code-block:: python

            title_context_template = "template"

            openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
            entity_extractor = OpenAIEntityExtractor("title", llm=openai_llm, prompt_template=title_context_template)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner())
                .extract_entity(entity_extractor=entity_extractor)


    """

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

    def extract_entity(self, document: Document) -> Document:
        if self._prompt_template:
            entities = self._handle_few_shot_prompting(document)
        else:
            entities = self._handle_zero_shot_prompting(document)

        document.properties.update({f"{self._entity_name}": entities["answer"]})

        return document

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
    """
    ExtractEntity is a transformation class for extracting entities from a dataset using an EntityExtractor.

    The Extract Entity Transform extracts semantically meaningful information from your documents.These extracted
    entities are then incorporated as properties into the document structure.

    Args:
        child: The source node or component that provides the dataset containing text data.
        entity_extractor: An instance of an EntityExtractor class that defines the entity extraction method to be
        applied.
        resource_args: Additional resource-related arguments that can be passed to the extraction operation.

    Example:
         .. code-block:: python

            source_node = ...  # Define a source node or component that provides a dataset with text data.
            custom_entity_extractor = MyEntityExtractor(entity_extraction_params)
            extraction_transform = ExtractEntity(child=source_node, entity_extractor=custom_entity_extractor)
            extracted_entities_dataset = extraction_transform.execute()

    """

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
        dataset = input_dataset.map(generate_map_function(self._entity_extractor.extract_entity))
        return dataset
