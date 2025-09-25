from typing import Optional
import ast
import json
import sycamore
from sycamore.docset import DocSet
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.llms.config import LLMModel
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import SycamorePrompt, RenderedPrompt, RenderedMessage
from sycamore.schema import SchemaV2
from sycamore.transforms.property_extraction.extract import SchemaExtract
from sycamore.transforms.property_extraction.strategy import BatchElements
from sycamore.transforms.property_extraction.merge_schemas import intersection_of_fields


class FakeExtractionPrompt(SycamorePrompt):
    def render_multiple_elements(self, elts: list[Element], doc: Document) -> RenderedPrompt:
        schema = doc.properties.get("_schema_temp")
        if (
            not schema
        ):  # The SchemaExtract init method calls this method with an empty Document so we need to handle that case
            return RenderedPrompt(messages=[])
        return RenderedPrompt(
            messages=[RenderedMessage(role="user", content=f"property={property}") for property in schema]
        )


class FakeLLM(LLM):
    def __init__(self):
        super().__init__(model_name="fake", default_mode=LLMMode.ASYNC)

    def is_chat_mode(self):
        return True

    def generate(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        ret_val = [ast.literal_eval(msg.content[9:]) for msg in prompt.messages]
        return f"""{json.dumps(ret_val)}"""

    async def generate_async(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        return self.generate(prompt=prompt, llm_kwargs=llm_kwargs)


class TestSchemaExtract:
    def test_schema_extract(self):
        doc_0 = Document(
            doc_id="0", elements=[Element(text_representation="d0e0"), Element(text_representation="d0e1")]
        )
        doc_0.properties["_schema_temp"] = [
            {
                "name": "company_name",
                "type": {
                    "type": "string",
                    "description": "Name of the company",
                    "examples": ["Acme Corp"],
                },
            },
            {
                "name": "ceo",
                "type": {
                    "type": "string",
                    "description": "CEO name",
                    "examples": ["Jane Doe"],
                },
            },
        ]

        doc_1 = Document(
            doc_id="1", elements=[Element(text_representation="d1e0"), Element(text_representation="d1e1")]
        )
        doc_1.properties["_schema_temp"] = [
            {
                "name": "company_name",
                "type": {
                    "type": "string",
                    "description": "Name of the company",
                    "examples": ["Beta LLC"],
                },
            },
            {
                "name": "revenue",
                "type": {
                    "type": "float",
                    "description": "Annual revenue",
                    "examples": [1000000.0],
                },
            },
        ]

        agg_schema_true = SchemaV2(
            properties=[
                {
                    "name": "company_name",
                    "type": {
                        "type": "string",
                        "description": "Name of the company",
                        "examples": ["Beta LLC", "Acme Corp"],
                    },
                }
            ]
        )

        docs = [doc_0, doc_1]
        context = sycamore.init()
        read_ds = context.read.document(docs)
        schema_ext = SchemaExtract(
            read_ds.plan,
            step_through_strategy=BatchElements(batch_size=50),
            llm=FakeLLM(),
            prompt=FakeExtractionPrompt(),
        )
        ds = DocSet(context, schema_ext).reduce(intersection_of_fields)
        agg_schema_pred = ds.take()[0].properties.get("_schema", SchemaV2(properties=[]))

        assert (
            agg_schema_pred.properties[0].name == agg_schema_true.properties[0].name
        ), f"Expected name {agg_schema_true.properties[0].name}, got {agg_schema_pred.properties[0].name}"
        assert (
            agg_schema_pred.properties[0].type.type.value == agg_schema_true.properties[0].type.type.value
        ), f"Expected type {agg_schema_true.properties[0].type.type.value}, got {agg_schema_pred.properties[0].type.type.value}"
        assert (
            agg_schema_pred.properties[0].type.description == agg_schema_true.properties[0].type.description
        ), f"Expected description {agg_schema_true.properties[0].type.description}, got {agg_schema_pred.properties[0].type.description}"
        assert set(agg_schema_pred.properties[0].type.examples) == set(
            agg_schema_true.properties[0].type.examples
        ), f"Expected examples {agg_schema_true.properties[0].type.examples}, got {agg_schema_pred.properties[0].type.examples}"
