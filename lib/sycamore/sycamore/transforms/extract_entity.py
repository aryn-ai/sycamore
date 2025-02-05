from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Union, cast

from sycamore.context import Context, context_params, OperationTypes
from sycamore.data import Element, Document
from sycamore.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    EntityExtractorZeroShotGuidancePrompt,
    EntityExtractorFewShotGuidancePrompt,
    _EntityExtractorZeroShotGuidancePrompt,
    _EntityExtractorFewShotGuidancePrompt,
)
from sycamore.llms.prompts.prompts import (
    ElementListIterPrompt,
    ElementListPrompt,
    RenderedMessage,
    SycamorePrompt,
    RenderedPrompt,
)
from sycamore.plan_nodes import Node
from sycamore.transforms.base_llm import LLMMap
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
from sycamore.functions.tokenizer import Tokenizer
from sycamore.transforms.similarity import SimilarityScorer
from sycamore.utils.similarity import make_element_sorter_fn
from sycamore.utils.llm_utils import merge_elements


def element_list_formatter(elements: list[Element], field: str = "text_representation") -> str:
    query = ""
    for i in range(len(elements)):
        value = str(elements[i].field_to_value(field))
        query += f"ELEMENT {i + 1}: {value}\n"
    return query


class FieldToValuePrompt(SycamorePrompt):
    def __init__(self, messages: list[RenderedMessage], field: str):
        self.messages = messages
        self.field = field

    def render_document(self, doc: Document) -> RenderedPrompt:
        value = doc.field_to_value(self.field)
        rendered = []
        for m in self.messages:
            rendered.append(RenderedMessage(role=m.role, content=m.content.format(value=value)))
        return RenderedPrompt(messages=rendered)


class EntityExtractor(ABC):
    def __init__(self, entity_name: str):
        self._entity_name = entity_name

    @abstractmethod
    def as_llm_map(
        self, child: Optional[Node], context: Optional[Context] = None, llm: Optional[LLM] = None, **kwargs
    ) -> Node:
        pass

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
        max_tokens: int = 512,
        tokenizer: Optional[Tokenizer] = None,
        similarity_query: Optional[str] = None,
        similarity_scorer: Optional[SimilarityScorer] = None,
    ):
        super().__init__(entity_name)
        self._entity_name = entity_name
        self._llm = llm
        self._num_of_elements = num_of_elements
        self._prompt_template = prompt_template
        self._prompt_formatter = prompt_formatter
        self._use_elements = use_elements
        self._prompt = prompt
        self._field = field
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._similarity_query = similarity_query
        self._similarity_scorer = similarity_scorer

    @context_params(OperationTypes.INFORMATION_EXTRACTOR)
    def as_llm_map(
        self, child: Optional[Node], context: Optional[Context] = None, llm: Optional[LLM] = None, **kwargs
    ) -> Node:
        if llm is None:
            llm = self._llm
        assert llm is not None, "Could not find an LLM to use"
        prompt: SycamorePrompt  # grr mypy
        if self._prompt_template is not None:
            prompt = EntityExtractorFewShotGuidancePrompt
            prompt = cast(ElementListPrompt, prompt.set(examples=self._prompt_template))
        else:
            prompt = EntityExtractorZeroShotGuidancePrompt

        if self._tokenizer is not None:

            def validate(d: Document) -> bool:
                return d.properties.get(self._entity_name, "None") != "None"

            def elt_list_ctor(elts: list[Element]) -> str:
                if self._prompt_formatter is not element_list_formatter:
                    return self._prompt_formatter(elts, self._field)
                combined_text = ""
                for element in elts:
                    if "type" in element:
                        combined_text += f"Element type: {element['type']}\n"
                    if "page_number" in element["properties"]:
                        combined_text += f"Page_number: {element['properties']['page_number']}\n"
                    if "_element_index" in element["properties"]:
                        combined_text += f"Element_index: {element['properties']['_element_index']}\n"
                    combined_text += f"Text: {element.field_to_value(self._field)}\n"
                return combined_text

            source_idx_key = f"{self._entity_name}_source_element_index"

            def eb(elts: list[Element]) -> list[list[Element]]:
                curr_tks = 0
                curr_batch: list[Element] = []
                batches = []
                source_indices = set()
                assert (
                    self._tokenizer is not None
                ), "Cannot batch elements based on token counts because tokenizer is None"
                for e in elts:
                    eltl = cast(ElementListPrompt, prompt).element_list_constructor([e])
                    tks = len(self._tokenizer.tokenize(eltl))
                    if tks + curr_tks > self._max_tokens:
                        batches.append(curr_batch)
                        curr_tks = tks
                        curr_batch = [e]
                        source_indices = {e.element_index}
                        e.properties[source_idx_key] = source_indices
                    else:
                        e.properties[source_idx_key] = source_indices
                        source_indices.add(e.element_index)
                        curr_batch.append(e)
                        curr_tks += tks
                batches.append(curr_batch)
                return batches

            iteration_var_name = f"{self._entity_name}_i"

            def postprocess(d: Document) -> Document:
                last_eclub: set[int] = set()
                club_idx = 0
                target_club_idx = d.properties[iteration_var_name]
                for e in d.elements:
                    if len(last_eclub) > 0 and e.properties[source_idx_key] != last_eclub:
                        club_idx += 1
                    last_eclub = e.properties[source_idx_key]
                    if club_idx == target_club_idx:
                        d.properties[source_idx_key] = last_eclub
                        break
                return d

            prompt = ElementListIterPrompt(
                system=prompt.system,
                user=prompt.user,
                element_list_constructor=elt_list_ctor,
                element_batcher=eb,
                entity=self._entity_name,
                examples=self._prompt_template,
                iteration_var_name=iteration_var_name,
            )

            llm_map = LLMMap(
                child, prompt, self._entity_name, llm, iteration_var=iteration_var_name, validate=validate, **kwargs
            )
            ppmap = Map(llm_map, f=postprocess)
            return ppmap

        elif not self._use_elements:
            if self._prompt is None:
                raise ValueError("prompt must be specified if use_elements is False")
            if isinstance(self._prompt, str):
                prompt = FieldToValuePrompt(
                    messages=[RenderedMessage(role="user", content=self._prompt + "{value}")], field=self._field
                )
            elif isinstance(self._prompt, list):
                ms = [RenderedMessage(role=m["role"], content=m["content"]) for m in self._prompt]
                ms.append(RenderedMessage(role="user", content="{value}"))
                prompt = FieldToValuePrompt(messages=ms, field=self._field)
            return LLMMap(child, prompt, self._entity_name, llm, **kwargs)

        def elt_sorter(elts: list[Element]) -> list[Element]:
            sorter_inner = make_element_sorter_fn(self._field, self._similarity_query, self._similarity_scorer)
            dummy_doc = Document(elements=elts)
            sorter_inner(dummy_doc)
            return dummy_doc.elements

        prompt = prompt.set(element_select=lambda e: elt_sorter(e)[: self._num_of_elements])
        prompt = prompt.set(element_list_constructor=lambda e: self._prompt_formatter(e, self._field))
        prompt = prompt.set(entity=self._entity_name)

        llm_map = LLMMap(child, prompt, self._entity_name, llm, **kwargs)
        return llm_map

    @context_params(OperationTypes.INFORMATION_EXTRACTOR)
    @timetrace("OaExtract")
    def extract_entity(
        self, document: Document, context: Optional[Context] = None, llm: Optional[LLM] = None
    ) -> Document:
        self._llm = llm or self._llm
        if self._use_elements:
            element_sorter = make_element_sorter_fn(self._field, self._similarity_query, self._similarity_scorer)
            element_sorter(document)
            if self._tokenizer is not None:
                entities, window_indices = self._handle_element_chunking(document)
                document.properties[f"{self._entity_name}_source_element_index"] = window_indices
            else:
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
                prompt = _EntityExtractorFewShotGuidancePrompt()
            else:
                prompt = _EntityExtractorZeroShotGuidancePrompt()
            entities = self._llm.generate_old(
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

    def _handle_element_chunking(self, document: Document) -> Any:
        assert self._tokenizer is not None
        ind = 0
        while ind < len(document.elements):
            ind, combined_text, window_indices = merge_elements(
                ind, document.elements, self._field, self._tokenizer, self._max_tokens
            )
            entity = self._get_entities(combined_text)
            for i in range(0, len(window_indices)):
                document.elements[ind - i - 1]["properties"][f"{self._entity_name}"] = entity
                document.elements[ind - i - 1]["properties"][
                    f"{self._entity_name}_source_element_index"
                ] = window_indices
            if entity != "None":
                return entity, window_indices
        return "None", None

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
            response = self._llm.generate_old(prompt_kwargs={"prompt": prompt}, llm_kwargs={})
        else:
            messages = (self._prompt or []) + [{"role": "user", "content": content}]
            response = self._llm.generate_old(prompt_kwargs={"messages": messages}, llm_kwargs={})
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
