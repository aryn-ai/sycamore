from typing import Optional, Any
import asyncio
import logging

from sycamore.data.document import Document
from sycamore.plan_nodes import Node
from sycamore.schema import Schema
from sycamore.transforms.map import MapBatch
from sycamore.transforms.property_extraction.strategy import SchemaPartitionStrategy, StepThroughStrategy
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.prompts import SycamorePrompt
from sycamore.utils.extract_json import extract_json
from sycamore.utils.threading import run_coros_threadsafe

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
        put_in_properties_dot_entity: bool = True,
    ):
        if put_in_properties_dot_entity:
            _logger.warning("Extraction results will go in properties.entity")
        super().__init__(node, f=self.extract)
        self._schema = schema
        self._step_through = step_through_strategy
        self._schema_partition = schema_partition_strategy
        self._llm = llm
        self._prompt = prompt
        # Try calling the render method I need at constructor to make sure it's implemented
        self._prompt.render_multiple_elements(elts=[], doc=Document())
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

    async def extract_schema_partition(self, documents: list[Document], schema_part: Schema) -> list[dict[str, Any]]:
        coros = [self.extract_schema_partition_from_document(d, schema_part) for d in documents]
        return await asyncio.gather(*coros)

    async def extract_schema_partition_from_document(self, document: Document, schema_part: Schema) -> dict[str, Any]:
        prompt = self._prompt.fork(schema=schema_part)
        expected_fields = sorted(f.name for f in schema_part.fields)
        result_dict = dict()
        for elements in self._step_through.step_through(document):
            rendered = prompt.render_multiple_elements(elements, document)
            result = await self._llm.generate_async(prompt=rendered)
            rd = extract_json(result)
            for k, v in rd.items():
                if v is not None and k not in result_dict and k in expected_fields:
                    result_dict[k] = v
            if sorted(result_dict.keys()) == expected_fields:
                return result_dict
        for sf in schema_part.fields:
            if sf.default is not None and sf.name not in result_dict:
                result_dict[sf.name] = sf.default
        return result_dict
