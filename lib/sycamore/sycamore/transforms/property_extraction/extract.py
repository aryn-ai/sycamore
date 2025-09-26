from typing import Optional, Any
import asyncio
import logging
import json

from sycamore.data.document import Document, Element
from sycamore.plan_nodes import Node
from sycamore.schema import NamedProperty, ObjectProperty, SchemaV2 as Schema, DataType
from sycamore.transforms.map import MapBatch
from sycamore.transforms.property_extraction.strategy import (
    SchemaPartitionStrategy,
    SchemaUpdateStrategy,
    SchemaUpdateResult,
    StepThroughStrategy,
    TakeFirstTrimSchema,
)
from sycamore.transforms.property_extraction.types import AttributionValue, RichProperty
from sycamore.transforms.property_extraction.prompts import schema_extract_pre_elements_helper, ExtractionJinjaPrompt
from sycamore.transforms.property_extraction.utils import remove_keys_recursive
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.prompts import SycamorePrompt
from sycamore.utils.extract_json import extract_json
from sycamore.utils.threading import run_coros_threadsafe
from sycamore.transforms.property_extraction.utils import stitch_together_objects, dedup_examples
from sycamore.transforms.property_extraction.attribution import refine_attribution
from sycamore.transforms.property_extraction.prompts import format_schema_v2
from sycamore.utils.zip_traverse import zip_traverse

_logger = logging.getLogger(__name__)

MAX_RETRIES = 3


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
            # Fill in missing null values
            for k, (prop, val), (prop_p, val_p) in zip_traverse(
                self._schema.as_object_property(),
                RichProperty(type=DataType.OBJECT, name=None, value=em),
                order="before",
                intersect_keys=False,
            ):
                # No value for this property, parent value exists,
                # This corresponds to a prop in the schema, and also don't add Nones to arrays
                if val is None and val_p is not None and prop is not None and prop_p.get_type() is not DataType.ARRAY:
                    dt = prop.get_type()
                    name = k if isinstance(k, str) else None
                    v: Any = [] if dt is DataType.ARRAY else ({} if dt is DataType.OBJECT else None)
                    sp = RichProperty(name=name, value=v, type=dt)
                    val_p._add_subprop(sp)
            # Copy the rich properties to 'plain' python values in prop.entity (and dump the rich prop)
            doc.properties.setdefault("entity", {})
            for k, v in em.items():
                if isinstance(v, RichProperty):
                    doc.properties["entity"][k] = v.to_python()
                    if not self._output_pydantic:
                        em[k] = v.model_dump()
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
        sch: Optional[Schema] = schema_part
        retries = 0

        working_results = RichProperty(type=DataType.OBJECT, value={}, name=None)
        while sch is not None and retries < MAX_RETRIES:
            sch_str = format_schema_v2(sch, working_results)
            prompt = self._prompt.fork(schema=sch_str)

            rendered = prompt.render_multiple_elements(elements, document)
            result = await self._llm.generate_async(prompt=rendered)
            rd = extract_json(result)
            rp = RichProperty.from_prediction(rd)

            for k, (v_new, v_work, prop), (p_new, p_work, prop_p) in zip_traverse(
                rp, working_results, sch.as_object_property(), order="before"
            ):
                if prop is None:
                    # If I didn't ask for this property, skip
                    continue
                if v_new is v_work:
                    # When I replace a list sub-items are the same, so skip
                    continue
                if v_new is not None and v_new.type is not DataType.OBJECT:
                    p_work.value[k] = v_new
                if v_new is not None and v_new.type is DataType.OBJECT:
                    p_work.value[k] = RichProperty(
                        name=k if isinstance(k, str) else None, type=DataType.OBJECT, value={}
                    )

            sch = self.validate_prediction(sch, working_results)
            retries += 1

        for k, (v,), (p,) in zip_traverse(working_results):
            v.attribution = AttributionValue(
                element_indices=[e.element_index if e.element_index is not None else -1 for e in elements]
            )
        working_results = refine_attribution(working_results, document)
        update = self._schema_update.update_schema(
            in_schema=schema_part, new_fields=working_results.value, existing_fields=result_dict
        )
        return update

    def validate_prediction(self, schema_part: Schema, prediction: RichProperty) -> Optional[Schema]:
        out_sch_obj = schema_part.model_copy(deep=True).as_object_property()
        # Inside array properties the same validator will be used multiple
        # times, but we only want to decrement it once per validate_prediction call
        decremented_validators = set()

        prop_to_inner_validators = dict()
        for k, (prop,), (prop_p,) in zip_traverse(out_sch_obj, order="after"):
            prop = prop.type if isinstance(prop, NamedProperty) else prop
            # Copy because references are weird
            prop_to_inner_validators[id(prop)] = [v for v in prop.validators]
            if prop.type is DataType.ARRAY:
                prop_to_inner_validators[id(prop)] += [v for v in prop_to_inner_validators[id(prop.item_type)]]
            if prop.type is DataType.OBJECT:
                for p in prop.properties:
                    prop_to_inner_validators[id(prop)] += [v for v in prop_to_inner_validators[id(p.type)]]

        for k, (val, prop), (val_p, prop_p) in zip_traverse(
            prediction, out_sch_obj, intersect_keys=True, order="after"
        ):
            # Don't try to validate an explicitly null prediction
            if val is None:
                continue
            prop = prop.type if isinstance(prop, NamedProperty) else prop
            for validator in prop.validators:
                valid, propval = validator.validate_property(val.to_python())
                val.is_valid = valid
                if val.type not in (DataType.ARRAY, DataType.OBJECT):
                    val.value = propval
                if not val.is_valid:
                    if id(validator) not in decremented_validators:
                        validator.n_retries -= 1
                        decremented_validators.add(id(validator))
                    val.invalid_guesses.append(val.value)
            if val.type is DataType.OBJECT:
                val.is_valid = all(inner.is_valid for inner in val.value.values())
            if val.type is DataType.ARRAY:
                val.is_valid = all(inner.is_valid for inner in val.value)

        pred_copy = prediction.model_copy(deep=True)

        for k, (val, prop), (val_p, prop_p) in zip_traverse(
            pred_copy, out_sch_obj, intersect_keys=False, order="before"
        ):
            if prop is None:
                continue
            oprop = prop
            prop = prop.type if isinstance(prop, NamedProperty) else prop
            trim = (
                val is None
                or len(prop_to_inner_validators[id(prop)]) == 0
                or val.is_valid
                or any(v.n_retries <= 0 for v in prop_to_inner_validators[id(prop)])
            )
            if val is not None and val.type is DataType.ARRAY:
                # Hack to prevent trimming properties inside arrays
                # by telling zip_traverse there's nothing to traverse.
                # I can get away with this bc I copied the prediction.
                val.value = []
            if trim and prop_p.get_type() is DataType.OBJECT:
                if isinstance(prop_p, NamedProperty):
                    prop_p = prop_p.type
                assert isinstance(prop_p, ObjectProperty), "Unreachable, type narrowing"
                prop_p.properties.remove(oprop)

        if len(out_sch_obj.properties) > 0:
            return Schema(properties=out_sch_obj.properties)
        return None


class SchemaExtract(MapBatch):
    def __init__(
        self,
        node: Optional[Node],
        *,
        step_through_strategy: StepThroughStrategy,
        llm: LLM,
        prompt: ExtractionJinjaPrompt,
        existing_schema: Optional[Schema] = None,
    ):
        super().__init__(node, f=self.extract_schema)
        self._step_through = step_through_strategy
        self._llm = llm
        if existing_schema is not None and len(existing_schema.properties) > 0:
            user_pre_elements = (prompt.user_pre_elements or "") + schema_extract_pre_elements_helper.format(
                existing_schema=json.dumps(remove_keys_recursive(existing_schema.model_dump()["properties"]), indent=2)
            )
            self._prompt = prompt.fork(user_pre_elements=user_pre_elements)
        else:
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
