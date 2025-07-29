from typing import Optional
import ast
import json
import sycamore
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import SycamorePrompt, RenderedPrompt, RenderedMessage
from sycamore.schema import Schema, SchemaField
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
            messages=[RenderedMessage(role="user", content=f"field={field.model_dump()}") for field in schema.fields]
        )


class FakeLLM(LLM):
    def __init__(self):
        super().__init__(model_name="fake", default_mode=LLMMode.ASYNC)

    def is_chat_mode(self):
        return True

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        temp = [ast.literal_eval(msg.content[6:]) for msg in prompt.messages]
        ret_val = [
            {
                "name": ii["name"],
                "value": ii["examples"][0],
                "type": ii["field_type"],
                "description": ii["description"],
            }
            for ii in temp
        ]
        return f"""{json.dumps(ret_val)}"""

    async def generate_async(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        return self.generate(prompt=prompt, llm_kwargs=llm_kwargs)


class TestSchemaExtract:
    def test_schema_extract(self):
        doc_0 = Document(
            doc_id="0", elements=[Element(text_representation="d0e0"), Element(text_representation="d0e1")]
        )
        doc_0.properties["_schema_temp"] = Schema(
            fields=[
                SchemaField(
                    name="company_name", field_type="string", description="Name of the company", examples=["Acme Corp"]
                ),
                SchemaField(name="ceo", field_type="string", description="CEO name", examples=["Jane Doe"]),
            ]
        )
        doc_1 = Document(
            doc_id="1", elements=[Element(text_representation="d1e0"), Element(text_representation="d1e1")]
        )
        doc_1.properties["_schema_temp"] = Schema(
            fields=[
                SchemaField(
                    name="company_name", field_type="string", description="Name of the company", examples=["Beta LLC"]
                ),
                SchemaField(name="revenue", field_type="float", description="Annual revenue", examples=[1000000.0]),
            ]
        )

        agg_schema_true = Schema(
            fields=[
                SchemaField(
                    name="company_name",
                    field_type="string",
                    description="Name of the company",
                    examples=["Beta LLC", "Acme Corp"],
                )
            ]
        )

        docs = [doc_0, doc_1]
        schema_ext = SchemaExtract(
            None,
            step_through_strategy=BatchElements(batch_size=50),
            llm=FakeLLM(),
            prompt=FakeExtractionPrompt(),
        )
        extracted_docs = schema_ext.run(docs)
        context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
        ds = context.read.document(extracted_docs).reduce(intersection_of_fields)
        agg_schema_pred = ds.take()[0].properties.get("_schema", Schema(fields=[]))

        assert (
            agg_schema_pred.fields[0].name == agg_schema_true.fields[0].name
        ), f"Expected name {agg_schema_true.fields[0].name}, got {agg_schema_pred.fields[0].name}"
        assert (
            agg_schema_pred.fields[0].field_type == agg_schema_true.fields[0].field_type
        ), f"Expected type {agg_schema_true.fields[0].field_type}, got {agg_schema_pred.fields[0].field_type}"
        assert (
            agg_schema_pred.fields[0].description == agg_schema_true.fields[0].description
        ), f"Expected description {agg_schema_true.fields[0].description}, got {agg_schema_pred.fields[0].description}"
        assert set(agg_schema_pred.fields[0].examples) == set(
            agg_schema_true.fields[0].examples
        ), f"Expected examples {agg_schema_true.fields[0].examples}, got {agg_schema_pred.fields[0].examples}"
