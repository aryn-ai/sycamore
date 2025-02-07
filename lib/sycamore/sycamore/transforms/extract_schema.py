from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Union
import json
import textwrap
import copy

from sycamore.data import Element, Document
from sycamore.connectors.common import flatten_data
from sycamore.llms.prompts.prompts import ElementListPrompt
from sycamore.schema import Schema
from sycamore.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    _SchemaZeroShotGuidancePrompt,
)
from sycamore.llms.prompts import SycamorePrompt, RenderedPrompt
from sycamore.llms.prompts.prompts import _build_format_str
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.transforms.base_llm import LLMMap
from sycamore.utils.extract_json import extract_json
from sycamore.utils.time_trace import timetrace

import dateparser


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
    def __init__(
        self,
    ):  # properties: list[str]):
        # self._properties = properties
        pass

    @abstractmethod
    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        pass


class PropertyExtractionFromSchemaPrompt(ElementListPrompt):
    default_system = "You are given text contents from a document."
    default_user = textwrap.dedent(
        """\
    Extract values for the following fields:
    {schema}

    Document text:
    {doc_text}

    Don't return extra information.
    If you cannot find a value for a requested property, use the provided default or the value 'None'.
    Return your answers as a valid json dictionary that will be parsed in python.
    """
    )

    def __init__(self, schema: Schema):
        super().__init__(system=self.default_system, user=self.default_user)
        self.schema = schema
        self.kwargs["schema"] = self._format_schema(schema)

    @staticmethod
    def _format_schema(schema: Schema) -> str:
        text = ""
        for i, field in enumerate(schema.fields):
            text += f"{i} {field.name}: type={field.field_type}: default={field.default}\n"
            if field.description is not None:
                text += f"    {field.description}\n"
            if field.examples is not None:
                text += f"    Examples values: {field.examples}\n"
        return text

    def set(self, **kwargs) -> SycamorePrompt:
        if "schema" in kwargs:
            new = copy.deepcopy(self)
            new.schema = kwargs["schema"]
            kwargs["schema"] = self._format_schema(new.schema)
            return new.set(**kwargs)
        return super().set(**kwargs)

    def render_document(self, doc: Document) -> RenderedPrompt:
        rp = super().render_document(doc)
        rp.response_format = self.schema.model_dump()
        return rp


class PropertyExtractionFromDictPrompt(ElementListPrompt):
    def __init__(self, schema: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.schema = schema

    def render_document(self, doc: Document) -> RenderedPrompt:
        format_args = copy.deepcopy(self.kwargs)
        format_args["doc_text"] = doc.text_representation
        if self.schema is None:
            schema = doc.properties.get("_schema")
        else:
            schema = self.schema
        format_args["schema"] = schema
        if "entity" not in format_args:
            format_args["entity"] = doc.properties.get("_schema_class", "entity")
        flat_props = flatten_data(doc.properties, prefix="doc_property", separator="_")
        format_args.update(flat_props)
        format_args["elements"] = self._render_element_list_to_string(doc)
        if doc.text_representation is None:
            format_args["doc_text"] = format_args["elements"]

        messages = _build_format_str(self.system, self.user, format_args)
        result = RenderedPrompt(messages=messages, response_format=schema)
        return result


class LLMSchemaExtractor(SchemaExtractor):
    """
    The LLMSchemaExtractor uses the specified LLM object to extract a schema.

    Args:
        entity_name: A natural-language name of the class to be extracted (e.g. `Corporation`)
        llm: An instance of an LLM for text processing.
        num_of_elements: The number of elements to consider for schema extraction. Default is 10.
        prompt_formatter: A callable function to format prompts based on document elements.

    Example:
        .. code-block:: python

            openai = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
            schema_extractor=LLMSchemaExtractor("Corporation", llm=openai, num_of_elements=35)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=ArynPartitioner())
                .extract_schema(schema_extractor=schema_extractor)
    """

    def __init__(
        self,
        entity_name: str,
        llm: LLM,
        num_of_elements: int = 35,
        max_num_properties: int = 7,
        prompt_formatter: Callable[[list[Element]], str] = element_list_formatter,
    ):
        super().__init__(entity_name)
        self._llm = llm
        self._num_of_elements = num_of_elements
        self._prompt_formatter = prompt_formatter
        self._max_num_properties = max_num_properties

    @timetrace("ExtrSchema")
    def extract_schema(self, document: Document) -> Document:
        entities = self._handle_zero_shot_prompting(document)

        try:
            payload = entities
            answer = extract_json(payload)
        except (json.JSONDecodeError, ValueError):
            answer = entities

        document.properties.update({"_schema": answer, "_schema_class": self._entity_name})

        return document

    def _handle_zero_shot_prompting(self, document: Document) -> Any:
        sub_elements = [document.elements[i] for i in range((min(self._num_of_elements, len(document.elements))))]

        prompt = _SchemaZeroShotGuidancePrompt()

        entities = self._llm.generate_old(
            prompt_kwargs={
                "prompt": prompt,
                "entity": self._entity_name,
                "max_num_properties": self._max_num_properties,
                "query": self._prompt_formatter(sub_elements),
            }
        )

        return entities


class OpenAISchemaExtractor(LLMSchemaExtractor):
    """Alias for LLMSchemaExtractor for OpenAI models.

    Retained for backward compatibility.

    .. deprecated:: 0.1.25
    Use LLMSchemaExtractor instead.
    """

    pass


class LLMPropertyExtractor(PropertyExtractor):
    """
    The LLMPropertyExtractor uses an LLM to extract actual property values once
    a schema has been detected or provided.

    Args:
        llm: An instance of an LLM for text processing.
        schema_name: An optional natural-language name of the class to be extracted (e.g. `Corporation`)
            If not provided, will use the _schema_class property added by extract_schema.
        schema: An optional JSON-encoded schema, or Schema object to be used for property extraction.
            If not provided, will use the _schema property added by extract_schema.
        num_of_elements: The number of elements to consider for property extraction. Default is 10.
        prompt_formatter: A callable function to format prompts based on document elements.

    Example:
        .. code-block:: python

            schema_name = "AircraftIncident"
            schema = {"location": "string", "aircraft": "string", "date_and_time": "string"}

            openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
            property_extractor = LLMPropertyExtractor(
                llm=openai, schema_name=schema_name, schema=schema, num_of_elements=35
            )

            docs_with_schema = ...
            docs_with_schema = docs_with_schema.extract_properties(property_extractor=property_extractor)
    """

    def __init__(
        self,
        llm: LLM,
        schema_name: Optional[str] = None,
        schema: Optional[Union[dict[str, str], Schema]] = None,
        num_of_elements: int = 10,
        prompt_formatter: Callable[[list[Element]], str] = element_list_formatter,
    ):
        super().__init__()
        self._llm = llm
        self._schema_name = schema_name
        self._schema = schema
        self._num_of_elements = num_of_elements
        self._prompt_formatter = prompt_formatter

    def extract_docs(self, docs: list[Document]) -> list[Document]:
        jsonextract_node = self.as_llm_map(None)
        assert len(jsonextract_node.children) == 1
        llm_map_node = jsonextract_node.children[0]
        assert isinstance(jsonextract_node, Map)
        assert isinstance(llm_map_node, LLMMap)
        return [jsonextract_node.run(d) for d in llm_map_node.run(docs)]

    def cast_types(self, fields: dict) -> dict:
        assert self._schema is not None, "Schema must be provided for property standardization."
        assert isinstance(self._schema, Schema), "Schema object must be provided for property standardization."
        result: dict = {}

        type_cast_functions: dict[str, Callable] = {
            "int": int,
            "float": float,
            "str": str,
            "string": str,
            "bool": bool,
            "date": lambda x: dateparser.parse(x),
            "datetime": lambda x: dateparser.parse(x),
        }

        for field in self._schema.fields:
            value = fields.get(field.name)
            if value is None and field.default is None:
                result[field.name] = None
            else:
                result[field.name] = type_cast_functions.get(field.field_type, lambda x: x)(value)

        # Include additional fields not defined in the schema
        for key, value in fields.items():
            if key not in result:
                result[key] = value

        return result

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        prompt: SycamorePrompt  # mypy grr
        if isinstance(self._schema, Schema):
            prompt = PropertyExtractionFromSchemaPrompt(self._schema)
        else:
            prompt = PropertyExtractionFromDictPrompt(
                schema=self._schema,
                system="You are a helpful property extractor. You only return JSON.",
                user=textwrap.dedent(
                    """\
                    You are given a few text elements of a document. Extract JSON representing one entity of
                    class {entity} from the document. The class only has properties {schema}. Using
                    this context, FIND, FORMAT, and RETURN the JSON representing one {entity}.
                    Only return JSON as part of your answer. If no entity is in the text, return "None".
                    {doc_text}
                    """
                ),
            )
            if self._schema_name is not None:
                prompt = prompt.set(entity=self._schema_name)

        def parse_json_and_cast(d: Document) -> Document:
            entity_name = self._schema_name or "_entity"
            entitystr = d.properties.get(entity_name, "{}")
            endkey = self._schema_name or d.properties.get("_schema_class", "entity")
            try:
                entity = extract_json(entitystr)
            except (json.JSONDecodeError, AttributeError, ValueError):
                entity = entitystr

            # If LLM couldn't do extract we instructed it to say "None"
            # So handle that
            if entity == "None":
                entity = {}

            if isinstance(self._schema, Schema):
                entity = self.cast_types(entity)

            # If schema name wasn't provided we wrote stuff to a
            # temp "_entity" property
            if entity_name == "_entity":
                if endkey in d.properties:
                    d.properties[endkey].update(entity)
                else:
                    d.properties[endkey] = entity
                if "_entity" in d.properties:
                    d.properties.pop("_entity")
                return d
            d.properties[endkey] = entity
            return d

        llm_map = LLMMap(child, prompt, output_field=self._schema_name or "_entity", llm=self._llm, **kwargs)
        parse_map = Map(llm_map, f=parse_json_and_cast)
        return parse_map


class ExtractSchema(Map):
    """
    ExtractSchema is a transformation class for extracting schemas from documents using an SchemaExtractor.

    This method will extract a unique schema for each document in the DocSet independently. If the documents in the
    DocSet represent instances with a common schema, consider `ExtractBatchSchema` which will extract a common
    schema for all documents.

    The dataset is returned with an additional `_schema` property that contains JSON-encoded schema, if any
    is detected.

    Args:
        child: The source node or component that provides the dataset text for schema suggestion
        schema_extractor: An instance of an SchemaExtractor class that provides the schema extraction method
        resource_args: Additional resource-related arguments that can be passed to the extraction operation

    Example:
         .. code-block:: python

            custom_schema_extractor = ExampleSchemaExtractor(entity_extraction_params)

            documents = ...  # Define a source node or component that provides a dataset with text data.
            documents_with_schema = ExtractSchema(child=documents, schema_extractor=custom_schema_extractor)
            documents_with_schema = documents_with_schema.execute()
    """

    def __init__(self, child: Node, schema_extractor: SchemaExtractor, **resource_args):
        super().__init__(child, f=schema_extractor.extract_schema, **resource_args)


class OpenAIPropertyExtractor(LLMPropertyExtractor):
    """Alias for LLMPropertyExtractor for OpenAI models.

    Retained for backward compatibility.

    .. deprecated:: 0.1.25
    Use LLMPropertyExtractor instead.
    """

    pass


class ExtractBatchSchema(Map):
    """
    ExtractBatchSchema is a transformation class for extracting a schema from a dataset using an SchemaExtractor.
    This assumes all documents in the dataset share a common schema.

    If it is more appropriate to provide a unique schema for each document (such as in a hetreogenous PDF collection)
    consider using `ExtractSchema` instead.

    The dataset is returned with an additional `_schema` property that contains JSON-encoded schema, if any
    is detected. This schema will be the same for all elements of the dataest.

    Args:
        child: The source node or component that provides the dataset text for schema suggestion
        schema_extractor: An instance of an SchemaExtractor class that provides the schema extraction method
        resource_args: Additional resource-related arguments that can be passed to the extraction operation

    Example:
         .. code-block:: python

            custom_schema_extractor = ExampleSchemaExtractor(entity_extraction_params)

            documents = ...  # Define a source node or component that provides a dataset with text data.
            documents_with_schema = ExtractBatchSchema(child=documents, schema_extractor=custom_schema_extractor)
            documents_with_schema = documents_with_schema.execute()
    """

    def __init__(self, child: Node, schema_extractor: SchemaExtractor, **resource_args):
        # Must run on a single instance so that the cached calculation of the schema works
        resource_args["parallelism"] = 1
        # super().__init__(child, f=lambda d: d, **resource_args)
        super().__init__(child, f=ExtractBatchSchema.Extract, constructor_args=[schema_extractor], **resource_args)

    class Extract:
        def __init__(self, schema_extractor: SchemaExtractor):
            self._schema_extractor = schema_extractor
            self._schema: Optional[dict] = None

        def __call__(self, d: Document) -> Document:
            if self._schema is None:
                s = self._schema_extractor.extract_schema(d)
                self._schema = {"_schema": s.properties["_schema"], "_schema_class": s.properties["_schema_class"]}

            d.properties.update(self._schema)

            return d
