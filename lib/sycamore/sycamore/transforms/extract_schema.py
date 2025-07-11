from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, List
import json
import sycamore
import logging
from sycamore import ExecMode
from sycamore.data import Element, Document
from sycamore.schema import Schema
from sycamore.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    PropertiesZeroShotJinjaPrompt,
    PropertiesFromSchemaJinjaPrompt,
    SchemaZeroShotJinjaPrompt,
)
from sycamore.llms.prompts import SycamorePrompt
from sycamore.plan_nodes import Node
from sycamore.transforms.base import CompositeTransform
from sycamore.transforms.map import Map
from sycamore.transforms.base_llm import LLMMap
from sycamore.utils.extract_json import extract_json
from sycamore.utils.time_trace import timetrace
from sycamore.transforms.embed import Embedder
from sycamore.llms.prompts.default_prompts import MetadataExtractorJinjaPrompt
import math


def cluster_schema_json(schema: Schema, cluster_size: int, embedder: Optional[Embedder] = None) -> List[Document]:
    field_docs: List[Document] = []
    for fld in schema.fields:
        txt = f"Field: {fld.name}\nDescription: {fld.description or ''}"
        field_docs.append(Document(text_representation=txt, **fld.__dict__))

    ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
    embeddings = ctx.read.document(field_docs).embed(embedder)

    centroids = embeddings.kmeans(K=cluster_size or round(math.sqrt(len(schema.fields))), iterations=40)
    clds = embeddings.clustering(centroids, cluster_field_name="cluster")

    clusters_docs = clds.take_all()
    groups = {}
    for d in clusters_docs:
        cluster = d["cluster"].item() if hasattr(d["cluster"], "item") else d["cluster"]
        if cluster not in groups:
            groups[cluster] = Document()
        groups[cluster].elements.append(Element(**d))
    return list(groups.values())


def batch_schema_json(schema: Schema, batch_size: int) -> List[Document]:
    groups = {}
    for batch_num in range(batch_size):
        groups[batch_num] = Document()

    field_count = len(schema.fields)

    for field_num in range(field_count):
        batch = field_num % batch_size
        groups[batch].elements.append(Element(**schema.fields[field_num].__dict__))
    return list(groups.values())


def element_list_formatter(elements: list[Element]) -> str:
    query = ""
    for i in range(len(elements)):
        query += f"ELEMENT {i + 1}: {elements[i].text_representation}\n"
    return query


class SchemaExtractor(ABC):
    def __init__(self, entity_name: str):
        self._entity_name = entity_name

    @abstractmethod
    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        pass

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

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        prompt = SchemaZeroShotJinjaPrompt.fork(
            entity=self._entity_name,
            max_num_properties=self._max_num_properties,
            num_elements=self._num_of_elements,
            field="text_representation",
        )
        if self._prompt_formatter is not element_list_formatter:
            prompt = prompt.fork(prompt_formatter=self._prompt_formatter)

        def parse_json(doc: Document) -> Document:
            schemastr = doc.properties.get("_schema", "{}")
            try:
                schema = extract_json(schemastr)
            except (json.JSONDecodeError, AttributeError, ValueError):
                schema = schemastr
            doc.properties["_schema"] = schema
            doc.properties["_schema_class"] = self._entity_name
            return doc

        llm_map = LLMMap(child, prompt=prompt, output_field="_schema", llm=self._llm)
        json_map = Map(llm_map, f=parse_json)
        comptransform = CompositeTransform(child, [])  # type: ignore
        comptransform.nodes = [llm_map, json_map]
        return comptransform

    @timetrace("ExtrSchema")
    def extract_schema(self, document: Document) -> Document:
        comptransform = self.as_llm_map(None)
        assert isinstance(comptransform, CompositeTransform)
        return comptransform._local_process([document])[0]


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
        schema: Optional[Union[dict, Schema]] = None,
        num_of_elements: Optional[int] = None,
        prompt_formatter: Callable[[list[Element]], str] = element_list_formatter,
        metadata_extraction: bool = False,
        embedder: Optional[Embedder] = None,
        group_size: Optional[int] = None,
        clustering: bool = True,
    ):
        super().__init__()
        self._llm = llm
        self._schema_name = schema_name
        self._schema = schema
        self._num_of_elements = num_of_elements
        self._metadata_extraction = metadata_extraction
        self._prompt_formatter = prompt_formatter
        self._group_size = group_size
        self._embedder = embedder
        self._clustering = clustering

    def extract_docs(self, docs: list[Document]) -> list[Document]:
        jsonextract_node = self.as_llm_map(None)
        assert len(jsonextract_node.children) == 1
        llm_map_node = jsonextract_node.children[0]
        assert isinstance(jsonextract_node, Map)
        assert isinstance(llm_map_node, LLMMap)
        return [jsonextract_node.run(d) for d in llm_map_node.run(docs)]

    def cast_types(self, fields: dict) -> dict:
        import dateparser  # type: ignore # No type stubs available for 'dateparser'; ignoring for mypy

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
            "list": list,
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
        if self._metadata_extraction:
            assert isinstance(self._schema, Schema), "check format of schema passed"
            self._group_size = self._group_size or round(math.sqrt(len(self._schema.fields)))
            if self._clustering:
                clusters_docs = cluster_schema_json(
                    schema=self._schema, embedder=self._embedder, cluster_size=self._group_size
                )
            else:
                clusters_docs = batch_schema_json(schema=self._schema, batch_size=self._group_size)
            tmp_props: list[str] = []
            for idx, field_doc in enumerate(clusters_docs):
                schema = {}
                schema_name = f"_tmp_cluster_{idx}"
                tmp_props.append(schema_name)
                assert isinstance(field_doc, Document), "Expected field_doc to be a Document instance"
                for field in field_doc.elements:
                    schema[field["name"]] = {
                        "description": field["description"],
                        "type": field["field_type"],
                        "default": field.get("default"),
                        "examples": field.get("examples"),
                    }
                prompt = MetadataExtractorJinjaPrompt.fork(
                    entity_name=schema_name,
                    response_format=schema,
                    schema=schema,
                )
                child = LLMMap(child, prompt=prompt, output_field=schema_name, llm=self._llm, **kwargs)

            def _merge(d: Document) -> Document:
                merged_metadata: dict = {}
                merged_provenance: dict = {}
                for k in tmp_props:
                    temp_metadata = {}
                    temp_provenance = {}
                    part = d.properties.pop(k, "{}")
                    try:
                        if isinstance(part, str):
                            part_json = extract_json(part)
                            if isinstance(part_json, dict):
                                for k, v in part_json.items():
                                    if v:
                                        temp_metadata[k] = v[0]
                                        temp_provenance[k] = v[1]
                                    else:
                                        temp_metadata[k] = None
                            merged_metadata.update(temp_metadata)
                            merged_provenance.update(temp_provenance)
                    except json.JSONDecodeError:
                        logging.error(f"Failed to decode JSON for property '{k}': {part}")
                d.properties[self._schema_name or "_entity"] = merged_metadata
                d.properties[(self._schema_name or "_entity") + "_metadata"] = merged_provenance
                return d

            return Map(child, f=_merge)

        if isinstance(self._schema, Schema):
            prompt = PropertiesFromSchemaJinjaPrompt
            prompt = prompt.fork(schema=self._schema, response_format=self._schema.model_dump())
        else:
            prompt = PropertiesZeroShotJinjaPrompt
            if self._schema is not None:
                prompt = prompt.fork(schema=self._schema)

            if self._schema_name is not None:
                prompt = prompt.fork(entity=self._schema_name)
        if self._num_of_elements is not None:
            prompt = prompt.fork(num_elements=self._num_of_elements)
        if self._prompt_formatter is not element_list_formatter:
            prompt = prompt.fork(prompt_formatter=self._prompt_formatter)

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
