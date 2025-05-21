from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Union

from sycamore.context import Context, context_params, OperationTypes
from sycamore.data import Element, Document
from sycamore.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    EntityExtractorZeroShotJinjaPrompt,
    EntityExtractorFewShotJinjaPrompt,
    _EntityExtractorZeroShotGuidancePrompt,
    _EntityExtractorFewShotGuidancePrompt,
)
from sycamore.llms.prompts.prompts import (
    RenderedMessage,
    SycamorePrompt,
    RenderedPrompt,
    JinjaPrompt,
)
from sycamore.llms.prompts.jinja_fragments import (
    J_ELEMENT_BATCHED_LIST,
    J_ELEMENT_BATCHED_LIST_WITH_METADATA,
)
from sycamore.plan_nodes import Node
from sycamore.transforms.base import CompositeTransform, BaseMapTransform
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
        self,
        child: Optional[Node],
        context: Optional[Context] = None,
        llm: Optional[LLM] = None,
        **kwargs,
    ) -> Node:
        pass

    @abstractmethod
    def extract_entity(
        self,
        document: Document,
        context: Optional[Context] = None,
        llm: Optional[LLM] = None,
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
        entity_type: Optional[str] = None,
        llm: Optional[LLM] = None,
        prompt_template: Optional[str] = None,
        num_of_elements: int = 10,
        prompt_formatter: Callable[[list[Element], str], str] = element_list_formatter,
        use_elements: Optional[bool] = True,
        prompt: Optional[Union[list[dict], str, SycamorePrompt]] = None,
        field: str = "text_representation",
        max_tokens: int = 512,
        tokenizer: Optional[Tokenizer] = None,
        similarity_query: Optional[str] = None,
        similarity_scorer: Optional[SimilarityScorer] = None,
    ):
        super().__init__(entity_name)
        self._entity_name = entity_name
        self._entity_type = entity_type
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

    def _get_const_variables(self) -> dict[str, str]:
        # These kept popping up in various places across the transforms
        return {
            "similarity_field_name": f"{self._field}_similarity_score",
            "source_idx_key": f"{self._entity_name}_source_indices",
            "batch_key": f"{self._entity_name}_batches",
            "iteration_var_name": f"{self._entity_name}_i",
        }

    def _get_prompt(self) -> SycamorePrompt:
        # there's like a million paths to cover but I think I have
        # them all
        vars = self._get_const_variables()
        if self._prompt_formatter is not element_list_formatter:
            j_elements = "{{ formatter(doc.elements) }}"
        elif self._tokenizer is not None:
            j_elements = J_ELEMENT_BATCHED_LIST_WITH_METADATA
        else:
            j_elements = J_ELEMENT_BATCHED_LIST
        if not self._use_elements:
            if self._prompt is None:
                raise ValueError("prompt must be specified if use_elements is False")
            j_elements = "{{ doc.field_to_value(field) }}"

        common_params = {
            "field": self._field,
            "num_elements": self._num_of_elements,
            "batch_key": vars["batch_key"],
            "iteration_var": vars["iteration_var_name"],
            "entity": self._entity_name,
            "use_elements": self._use_elements,
        }

        if self._prompt is not None:
            if isinstance(self._prompt, SycamorePrompt):
                return self._prompt.fork(**common_params)
            if isinstance(self._prompt, str):
                return JinjaPrompt(
                    system=None,
                    user=self._prompt + "\n" + j_elements,
                    response_format=None,
                    **common_params,
                )
            else:

                system = None
                if len(self._prompt) > 0 and self._prompt[0]["role"] == "system":
                    system = self._prompt[0]["content"]
                    user = [p["content"] for p in self._prompt[1:]] + [j_elements]
                else:
                    user = [p["content"] for p in self._prompt] + [j_elements]
                return JinjaPrompt(system=system, user=user, response_format=None, **common_params)
        elif self._prompt_template is not None:
            return EntityExtractorFewShotJinjaPrompt.fork(examples=self._prompt_template, **common_params)
        else:
            return EntityExtractorZeroShotJinjaPrompt.fork(**common_params)

    def _make_preprocess_fn(self, prompt: SycamorePrompt) -> Callable[[Document], Document]:
        vars = self._get_const_variables()

        def sort_and_batch_elements(doc: Document) -> Document:
            if self._similarity_query is not None and self._similarity_scorer is not None:
                # If we did similarity scoring sort the elements (keep track of their original
                # locations though)
                elements = sorted(
                    [(e, i) for i, e in enumerate(doc.elements)],
                    key=(lambda e_i: e_i[0].properties.get(vars["similarity_field_name"], float("-inf"))),
                    reverse=True,
                )
            else:
                elements = [(e, i) for i, e in enumerate(doc.elements)]

            batches = []
            if self._tokenizer is not None:
                curr_club = []
                # We'll create a dummy document and consecutively
                # add more elements to it, rendering out to a prompt
                # at each step and counting tokens to find breakpoints.
                dummy = doc.copy()
                dummy.properties = doc.properties.copy()
                dummy.properties[vars["iteration_var_name"]] = 0
                dummy.elements = []
                for e, i in elements:
                    dummy.elements.append(e)
                    curr_club.append(i)
                    dummy.properties[vars["batch_key"]] = [curr_club]
                    rendered = prompt.render_document(dummy)
                    tks = rendered.token_count(self._tokenizer)
                    if tks > self._max_tokens:
                        curr_club.pop()
                        if len(curr_club) > 0:
                            batches.append(curr_club)
                        curr_club = [i]
                        e.properties[vars["source_idx_key"]] = curr_club
                        # dummy.elements = [e]
                    else:
                        e.properties[vars["source_idx_key"]] = curr_club
                if len(curr_club) > 0:
                    batches.append(curr_club)
            else:
                # If no tokenizer, we run a single batch with the first num_of_elements.
                batches = [[i for e, i in elements[: self._num_of_elements]]]
                for i in batches[0]:
                    doc.elements[i].properties[vars["source_idx_key"]] = batches[0]

            doc.properties[vars["batch_key"]] = batches
            return doc

        return sort_and_batch_elements

    @context_params(OperationTypes.INFORMATION_EXTRACTOR)
    def as_llm_map(
        self,
        child: Optional[Node],
        context: Optional[Context] = None,
        llm: Optional[LLM] = None,
        **kwargs,
    ) -> Node:
        # represent this EntityExtractor as a CompositeTransform consisting of some
        # preprocessing (set up batches, sort elements, etc), the central LLMMap,
        # and some postprocessing (derive the source_indices property)
        if llm is None:
            llm = self._llm
        assert llm is not None, "Could not find an LLM to use"

        prompt = self._get_prompt()
        preprocess = self._make_preprocess_fn(prompt)
        vars = self._get_const_variables()

        def validate(d: Document) -> bool:
            return self._tokenizer is None or d.properties.get(self._entity_name, "None") != "None"

        def postprocess(d: Document) -> Document:
            target_club_idx = d.properties[vars["iteration_var_name"]]
            if target_club_idx >= len(d.properties[vars["batch_key"]]):
                return d
            batch = d.properties[vars["batch_key"]][target_club_idx]
            d.properties[vars["source_idx_key"]] = batch
            if d.properties[self._entity_name] == "None":
                d.properties[self._entity_name] = None
            elif self._entity_type is not None and self._entity_type in [
                "int",
                "float",
            ]:
                try:
                    conversion_func = {"int": int, "float": float}[self._entity_type]
                    d.properties[self._entity_name] = conversion_func(d.properties[self._entity_name])
                except ValueError:
                    d.properties[self._entity_name] = None
            return d

        nodes: list[BaseMapTransform] = []
        head_node: Node
        if self._similarity_query is not None and self._similarity_scorer is not None:
            # If similarity we add a ScoreSimilarity node to the sub-pipeline
            from sycamore.transforms.similarity import ScoreSimilarity

            head_node = ScoreSimilarity(
                child,  # type: ignore
                similarity_scorer=self._similarity_scorer,
                query=self._similarity_query,
                score_property_name=vars["similarity_field_name"],
            )
            nodes.append(head_node)
        else:
            head_node = child  # type: ignore

        head_node = Map(head_node, f=preprocess)
        nodes.append(head_node)
        head_node = LLMMap(
            head_node,
            prompt,
            self._entity_name,
            llm,
            validate=validate,
            iteration_var=vars["iteration_var_name"],
            max_tries=100,
            **kwargs,
        )
        nodes.append(head_node)
        head_node = Map(head_node, f=postprocess)
        nodes.append(head_node)
        comptransform = CompositeTransform(child, [])  # type: ignore
        comptransform.nodes = nodes
        return comptransform

    @context_params(OperationTypes.INFORMATION_EXTRACTOR)
    @timetrace("OaExtract")
    def extract_entity(
        self,
        document: Document,
        context: Optional[Context] = None,
        llm: Optional[LLM] = None,
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
        assert not isinstance(
            self._prompt, SycamorePrompt
        ), f"cannot use old extract_entity interface with a SycamorePrompt: {self._prompt}"
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
        super().__init__(
            child,
            f=entity_extractor.extract_entity,
            kwargs={"context": context},
            **resource_args,
        )
