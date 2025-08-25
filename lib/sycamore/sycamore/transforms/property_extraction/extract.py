from typing import Optional, Any
import asyncio
import logging

from sycamore.data.document import Document, Element
from sycamore.plan_nodes import Node
from sycamore.schema import SchemaV2 as Schema, DataType
from sycamore.transforms.map import MapBatch
from sycamore.transforms.property_extraction.strategy import (
    SchemaPartitionStrategy,
    SchemaUpdateStrategy,
    SchemaUpdateResult,
    StepThroughStrategy,
    TakeFirstTrimSchema,
)
from sycamore.transforms.property_extraction.types import RichProperty
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.prompts import SycamorePrompt
from sycamore.utils.extract_json import extract_json
from sycamore.utils.threading import run_coros_threadsafe
from sycamore.transforms.property_extraction.utils import stitch_together_objects, dedup_examples
from sycamore.transforms.property_extraction.attribution import refine_attribution

_logger = logging.getLogger(__name__)


class Extract(MapBatch):

    def __init__(
        self,
        node: Optional[Node],
        *,
        schema: Schema,
        step_through_strategy: StepThroughStrategy,
        schema_partition_strategy: SchemaPartitionStrategy,
        llm: LLM,
        prompt: SycamorePrompt,
        schema_update_strategy: SchemaUpdateStrategy = TakeFirstTrimSchema(),
        output_pydantic_models: bool = True,
    ):
        super().__init__(node, f=self.extract)
        self._schema = schema
        self._step_through = step_through_strategy
        self._schema_partition = schema_partition_strategy
        self._schema_update = schema_update_strategy
        self._llm = llm
        self._prompt = prompt
        self._output_pydantic = output_pydantic_models
        # Try calling the render method I need at constructor to make sure it's implemented
        try:
            self._prompt.render_multiple_elements(elts=[], doc=Document())
        except NotImplementedError as e:
            raise e
        except Exception:
            # Other errors are ok, probably dummy document is malformed for the prompt
            pass

    def extract(self, documents: list[Document]) -> list[Document]:
        schema_parts = self._schema_partition.partition_schema(self._schema)
        coros = [self.extract_schema_partition(documents, sp) for sp in schema_parts]
        results = run_coros_threadsafe(coros)
        assert all(isinstance(r, dict) for result_set in results for r in result_set)
        for partial_result in results:
            for props, doc in zip(partial_result, documents):
                if "entity_metadata" not in doc.properties:
                    doc.properties["entity_metadata"] = props
                else:
                    meta = doc.properties["entity_metadata"]
                    rp = RichProperty(name=None, type=DataType.OBJECT, value=props)
                    rm = RichProperty(name=None, type=DataType.OBJECT, value=meta)
                    stitched = stitch_together_objects(rm, rp)
                    doc.properties["entity_metadata"] = stitched.value
        for doc in documents:
            em = doc.properties["entity_metadata"]
            doc.properties.setdefault("entity", {})
            for k, v in em.items():
                if isinstance(v, RichProperty):
                    doc.properties["entity"][k] = v.to_python()
                    if not self._output_pydantic:
                        em[k] = v.dump_recursive()
                else:
                    pass  # This property has already been added and de-pydanticized
        return documents

    async def extract_schema_partition(
        self, documents: list[Document], schema_part: Schema
    ) -> list[dict[str, RichProperty]]:
        coros = [self.extract_schema_partition_from_document(d, schema_part) for d in documents]
        return await asyncio.gather(*coros)

    async def extract_schema_partition_from_document(
        self, document: Document, schema_part: Schema
    ) -> dict[str, RichProperty]:
        em = document.properties.get("entity_metadata", {})
        result_dict = {k: RichProperty.validate_recursive(v) for k, v in em.items()}
        update = self._schema_update.update_schema(in_schema=schema_part, new_fields={}, existing_fields=result_dict)
        result_dict = update.out_fields
        schema_part = update.out_schema
        if update.completed:
            return result_dict

        for elements in self._step_through.step_through(document):
            update = await self.extract_schema_partition_from_element_batch(
                document, elements, schema_part, result_dict
            )
            result_dict = update.out_fields
            schema_part = update.out_schema
            if update.completed:
                return result_dict
        return result_dict

    async def extract_schema_partition_from_element_batch(
        self, document: Document, elements: list[Element], schema_part: Schema, result_dict: dict[str, RichProperty]
    ) -> SchemaUpdateResult:
        prompt = self._prompt.fork(schema=schema_part)

        rendered = prompt.render_multiple_elements(elements, document)
        result = await self._llm.generate_async(prompt=rendered)
        rd = extract_json(result)
        new_fields = dict()
        for k, v in rd.items():
            new_fields[k] = refine_attribution(RichProperty.from_prediction(v, elements, name=k), document)
        update = self._schema_update.update_schema(
            in_schema=schema_part, new_fields=new_fields, existing_fields=result_dict
        )
        return update


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
                doc.properties["_schema"] = Schema(properties=[])
                continue
            schema = Schema(properties=result)
            for prop in schema.properties:
                examples = dedup_examples(prop.type.examples or [])[:5]  # Limit to 5 examples
                prop.type.examples = examples if examples else None
            doc.properties["_schema"] = schema

        return documents

    async def extract_schema_from_document(self, document: Document) -> list[dict[str, Any]]:

        result_dict = dict()

        for elements in self._step_through.step_through(document):

            rendered = self._prompt.render_multiple_elements(elements, document)
            result = await self._llm.generate_async(prompt=rendered)
            rd = {ii["name"]: ii for ii in extract_json(result)}

            for k, v in rd.items():

                v_type = v.get("type", {}).get("type", "string")  # Default to string if type is not specified
                examples = v.get("type", {}).get("examples", None)

                if k not in result_dict:
                    v["type"]["examples"] = examples
                    v["type"]["type"] = v_type
                    result_dict[k] = v
                else:
                    # Only add examples if they match the type of the existing field
                    if examples is not None and v_type == result_dict[k]["type"]["type"]:
                        if result_dict[k]["type"]["examples"] is None:
                            result_dict[k]["type"]["examples"] = examples
                        else:
                            result_dict[k]["type"]["examples"].extend(examples)

                    elif v_type != result_dict[k]["type"]:
                        _logger.warning(
                            f"Type mismatch for field '{k}': {v_type} vs {result_dict[k]['type']['type']}. "
                            "Skipping this field."
                        )

        return list(result_dict.values())
