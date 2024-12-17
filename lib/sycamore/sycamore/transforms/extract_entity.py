from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Union

from sycamore.context import Context, context_params, OperationTypes
from sycamore.data import Element, Document
from sycamore.llms import LLM
from sycamore.llms.prompts import (
    EntityExtractorZeroShotGuidancePrompt,
    EntityExtractorFewShotGuidancePrompt,
)
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace


def element_list_formatter(elements: list[Element], field: str = "text_representation") -> str:
    query = ""
    for i in range(len(elements)):
        value = str(elements[i].field_to_value(field))
        query += f"ELEMENT {i + 1}: {value}\n"
    return query


class EntityExtractor(ABC):
    def __init__(self, entity_name: str):
        self._entity_name = entity_name

    @abstractmethod
    def extract_entity(
        self, document: Document, context: Optional[Context] = None, llm: Optional[LLM] = None
    ) -> Document:
        pass

    def property(self):
        """The name of the property added by calling extract_entity"""
        return self._entity_name


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
        llm: Optional[LLM] = None,
        prompt_template: Optional[str] = None,
        num_of_elements: int = 10,
        prompt_formatter: Callable[[list[Element], str], str] = element_list_formatter,
        use_elements: Optional[bool] = True,
        prompt: Optional[Union[list[dict], str]] = None,
        field: str = "text_representation",
    ):
        super().__init__(entity_name)
        self._llm = llm
        self._num_of_elements = num_of_elements
        self._prompt_template = prompt_template
        self._prompt_formatter = prompt_formatter
        self._use_elements = use_elements
        self._prompt = prompt
        self._field = field

    @context_params(OperationTypes.INFORMATION_EXTRACTOR)
    @timetrace("OaExtract")
    def extract_entity(
        self, document: Document, context: Optional[Context] = None, llm: Optional[LLM] = None
    ) -> Document:
        self._llm = llm or self._llm
        if self._use_elements:
            entities = self._handle_element_prompting(document)
        else:
            if self._prompt is None:
                raise Exception("prompt must be specified if use_elements is False")
            entities = self._handle_document_field_prompting(document)

        document.properties.update({f"{self._entity_name}": entities})

        return document

    def _handle_element_prompting(self, document: Document) -> Any:
        assert self._llm is not None
        sub_elements = [document.elements[i] for i in range((min(self._num_of_elements, len(document.elements))))]
        content = self._prompt_formatter(sub_elements, self._field)
        if self._prompt is None:
            prompt: Any = None
            if self._prompt_template:
                prompt = EntityExtractorFewShotGuidancePrompt()
            else:
                prompt = EntityExtractorZeroShotGuidancePrompt()
            entities = self._llm.generate(
                prompt_kwargs={
                    "prompt": prompt,
                    "entity": self._entity_name,
                    "query": content,
                    "examples": self._prompt_template,
                }
            )
            return entities
        else:
            return self._get_entities(content)

    def _handle_document_field_prompting(self, document: Document) -> Any:
        assert self._llm is not None
        if self._field is None:
            self._field = "text_representation"

        value = str(document.field_to_value(self._field))

        return self._get_entities(value)

    def _get_entities(self, content: str, prompt: Optional[Union[list[dict], str]] = None):
        assert self._llm is not None
        prompt = prompt or self._prompt
        assert prompt is not None, "No prompt found for entity extraction"
        if isinstance(self._prompt, str):
            prompt = self._prompt + content
            response = self._llm.generate(prompt_kwargs={"prompt": prompt}, llm_kwargs={})
        else:
            messages = (self._prompt or []) + [{"role": "user", "content": content}]
            response = self._llm.generate(prompt_kwargs={"messages": messages}, llm_kwargs={})
        return response


class ExtractEntity(Map):
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
        context: Optional[Context] = None,
        **resource_args,
    ):
        super().__init__(child, f=entity_extractor.extract_entity, kwargs={"context": context}, **resource_args)
