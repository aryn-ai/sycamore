from typing import Optional, Any
import asyncio
import logging

from sycamore.data.document import Document
from sycamore.plan_nodes import Node
from sycamore.schema import SchemaV2
from sycamore.transforms.map import MapBatch
from sycamore.transforms.property_extraction.strategy import (
    SchemaPartitionStrategy,
    SchemaUpdateStrategy,
    StepThroughStrategy,
    TakeFirstTrimSchema,
    RichProperty,
)
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.prompts import SycamorePrompt
from sycamore.utils.extract_json import extract_json
from sycamore.utils.threading import run_coros_threadsafe
from sycamore.transforms.property_extraction.utils import create_named_property

_logger = logging.getLogger(__name__)


class Extract(MapBatch):

    def __init__(
        self,
        node: Optional[Node],
        *,
        schema: SchemaV2,
        step_through_strategy: StepThroughStrategy,
        schema_partition_strategy: SchemaPartitionStrategy,
        llm: LLM,
        prompt: SycamorePrompt,
        put_in_properties_dot_entity: bool = True,
        schema_update_strategy: SchemaUpdateStrategy = TakeFirstTrimSchema(),
    ):
        if put_in_properties_dot_entity:
            _logger.warning("Extraction results will go in properties.entity")
        super().__init__(node, f=self.extract)
        self._schema = schema
        self._step_through = step_through_strategy
        self._schema_partition = schema_partition_strategy
        self._schema_update = schema_update_strategy
        self._llm = llm
        self._prompt = prompt
        # Try calling the render method I need at constructor to make sure it's implemented
        self._prompt.render_multiple_elements(elts=[], doc=Document(binary_representation=b""))
        self._pipde = put_in_properties_dot_entity

    def extract(self, documents: list[Document]) -> list[Document]:
        schema_parts = self._schema_partition.partition_schema(self._schema)
        coros = [self.extract_schema_partition(documents, sp) for sp in schema_parts]
        results = run_coros_threadsafe(coros)
        assert all(isinstance(r, dict) for result_set in results for r in result_set)
        for partial_result in results:
            for props, doc in zip(partial_result, documents):
                if self._pipde:
                    if "entity" not in doc.properties:
                        doc.properties["entity"] = props
                    else:
                        doc.properties["entity"].update(props)
                else:
                    doc.properties.update(props)

        return documents

    async def extract_schema_partition(self, documents: list[Document], schema_part: SchemaV2) -> list[dict[str, Any]]:
        coros = [self.extract_schema_partition_from_document(d, schema_part) for d in documents]
        return await asyncio.gather(*coros)

    async def extract_schema_partition_from_document(self, document: Document, schema_part: SchemaV2) -> dict[str, Any]:
        prompt = self._prompt.fork(schema=schema_part)
        result_dict: dict[str, RichProperty] = dict()
        for elements in self._step_through.step_through(document):
            rendered = prompt.render_multiple_elements(elements, document)
            result = await self._llm.generate_async(prompt=rendered)
            rd = extract_json(result)
            new_fields = dict()
            for k, v in rd.items():
                new_fields[k] = RichProperty.from_prediction(v, elements, name=k)
            update = self._schema_update.update_schema(
                in_schema=schema_part, new_fields=new_fields, existing_fields=result_dict
            )
            result_dict = update.out_fields
            schema_part = update.out_schema
            if update.completed:
                return result_dict
            prompt = self._prompt.fork(schema=schema_part)
        return result_dict


class SchemaExtract(MapBatch):
    def __init__(
        self,
        node: Optional[Node],
        *,
        step_through_strategy: StepThroughStrategy,
        llm: LLM,
        prompt: SycamorePrompt,
    ):
        super().__init__(node, f=self.extract_schema)
        self._step_through = step_through_strategy
        self._llm = llm
        self._prompt = prompt
        # Try calling the render method I need at constructor to make sure it's implemented
        self._prompt.render_multiple_elements(elts=[], doc=Document())

    @staticmethod
    def _cast_to_type(val: Any, val_type: str) -> Any:
        if isinstance(val, str):
            val = val.strip()
        if val is None or val in ["", "null", "None"]:
            return None
        conversion_f = {"int": int, "float": float}
        if val_type in conversion_f:
            try:
                return conversion_f[val_type](val)
            except ValueError:
                return None
        if val_type == "bool":
            try:
                return val.lower() == "true"
            except AttributeError:
                return None
        return val

    def extract_schema(self, documents: list[Document]) -> list[Document]:
        coros = [self.extract_schema_from_document(doc) for doc in documents]
        results = run_coros_threadsafe(coros)
        assert all(isinstance(r, list) for r in results)

        # Create one schema document per schema
        for result, doc in zip(results, documents):
            if not result:
                _logger.warning("No schema fields extracted, returning empty schema.")
                doc.properties["_schema"] = SchemaV2(properties=[])
                continue
            doc.properties["_schema"] = SchemaV2(properties=[create_named_property(prop) for prop in result])

        return documents

    async def extract_schema_from_document(self, document: Document) -> list[dict[str, Any]]:

        result_dict = dict()

        for elements in self._step_through.step_through(document):

            rendered = self._prompt.render_multiple_elements(elements, document)
            result = await self._llm.generate_async(prompt=rendered)
            rd = {ii["name"]: ii for ii in extract_json(result)}

            for k, v in rd.items():

                v_type = v.get("type", "string")  # Default to "string" if not specified
                example = self._cast_to_type(v.get("value", None), v_type)

                if k not in result_dict:
                    v["examples"] = [example] if example is not None else []
                    v["type"] = v_type
                    v.pop("value", None)
                    result_dict[k] = v
                else:
                    # Only append if type matches and value is not None and not already present
                    if example is not None and v_type == result_dict[k]["type"]:
                        if example not in result_dict[k]["examples"]:
                            result_dict[k]["examples"].append(example)
                    elif v_type != result_dict[k]["type"]:
                        _logger.warning(
                            f"Type mismatch for field '{k}': {v_type} vs {result_dict[k]['type']}. "
                            "Skipping this field."
                        )

        return list(result_dict.values())
