from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Union
import json

from sycamore.data import Element, Document
from sycamore.schema import Schema
from sycamore.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    _SchemaZeroShotGuidancePrompt,
    PropertiesZeroShotGuidancePrompt,
)
from sycamore.llms.prompts.default_prompts import ExtractPropertiesFromSchemaPrompt
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
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
    def extract_properties(self, document: Document) -> Document:
        pass


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

    @timetrace("ExtrProps")
    def extract_properties(self, document: Document) -> Document:
        entities = self._handle_zero_shot_prompting(document)

        try:
            payload = entities
            answer = extract_json(payload)
        except (json.JSONDecodeError, AttributeError):
            answer = entities
        if answer == "None":
            answer = {}

        if isinstance(self._schema, Schema):
            answer = self.cast_types(answer)
        if "entity" in document.properties:
            document.properties["entity"].update(answer)
        else:
            document.properties.update({"entity": answer})

        return document

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

    def _handle_zero_shot_prompting(self, document: Document) -> Any:
        if document.text_representation:
            text = document.text_representation
        else:
            text = self._prompt_formatter(
                [document.elements[i] for i in range((min(self._num_of_elements, len(document.elements))))]
            )
        if isinstance(self._schema, Schema):
            prompt = ExtractPropertiesFromSchemaPrompt(schema=self._schema, text=text)
            entities = self._llm.generate_old(prompt_kwargs={"prompt": prompt})
        else:
            schema = self._schema or document.properties.get("_schema")
            assert schema is not None, "Schema must be provided or detected before extracting properties."

            schema_name = self._schema_name or document.properties.get("_schema_class")
            assert schema_name is not None, "Schema name must be provided or detected before extracting properties."

            entities = self._llm.generate_old(
                prompt_kwargs={
                    "prompt": PropertiesZeroShotGuidancePrompt(),
                    "entity": schema_name,
                    "properties": schema,
                    "text": text,
                }
            )
        return entities


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


class ExtractProperties(Map):
    """
    ExtractProperties is a transformation class for extracting property values from a document once a schema has
    been established.

    The schema may be detected by `ExtractSchema` or provided manually under the `_schema` key of `Document.properties`.

    Args:
        child: The source node or component that provides the dataset text for schema suggestion
        property_extractor: An instance of an PropertyExtractor class that provides the property detection method
        resource_args: Additional resource-related arguments that can be passed to the extraction operation

    Example:
         .. code-block:: python

            documents = ...  # Define a source node or component that provides a dataset with text data.
            custom_property_extractor = ExamplePropertyExtractor(entity_extraction_params)

            documents_with_schema = ...
            documents_with_properties = ExtractProperties(
                child=documents_with_schema,
                property_extractor=custom_property_extractor
            )
            documents_with_properties = documents_with_properties.execute()
    """

    def __init__(self, child: Node, property_extractor: PropertyExtractor, **resource_args):
        super().__init__(child, f=property_extractor.extract_properties, **resource_args)
