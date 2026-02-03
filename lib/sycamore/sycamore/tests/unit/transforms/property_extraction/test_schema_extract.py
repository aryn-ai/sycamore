from typing import Optional
import ast
import json
import sycamore
from sycamore.docset import DocSet
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.llms.config import LLMModel
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import RenderedPrompt, RenderedMessage
from sycamore.transforms.property_extraction.prompts import ExtractionJinjaPrompt
from sycamore.schema import SchemaV2
from sycamore.transforms.property_extraction.extract import SchemaExtract
from sycamore.transforms.property_extraction.strategy import BatchElements
from sycamore.transforms.property_extraction.merge_schemas import (
    intersection_of_fields,
    union_of_fields,
    make_freq_filter_fn,
)


class FakeExtractionPrompt(ExtractionJinjaPrompt):
    def render_multiple_elements(self, elts: list[Element], doc: Document) -> RenderedPrompt:
        schema = doc.properties.get("_schema_temp")
        if (
            not schema
        ):  # The SchemaExtract init method calls this method with an empty Document so we need to handle that case
            return RenderedPrompt(messages=[])
        return RenderedPrompt(
            messages=[RenderedMessage(role="user", content=f"property={property}") for property in schema]
        )

    def fork(self, **kwargs):
        # Override fork to return an instance of the fake prompt. This is needed because the base class fork method
        # returns an instance of the base class, which we don't want.
        new_prompt = FakeExtractionPrompt()
        for k, v in kwargs.items():
            setattr(new_prompt, k, v)
        return new_prompt


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


class FlakyLLM(LLM):
    def __init__(self, max_fails: int):
        super().__init__(model_name="fake", default_mode=LLMMode.ASYNC)
        self.call_count = 0
        self.max_fails = max_fails

    def is_chat_mode(self):
        return True

    def generate(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        self.call_count += 1
        bad_output = '["properties": ['
        if self.max_fails < 0:
            return bad_output

        if self.call_count >= self.max_fails:
            ret_val = [ast.literal_eval(msg.content[9:]) for msg in prompt.messages]
            return f"""{json.dumps(ret_val)}"""

        return bad_output

    async def generate_async(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        return self.generate(prompt=prompt, llm_kwargs=llm_kwargs)


doc_0 = Document(doc_id="0", elements=[Element(text_representation="d0e0"), Element(text_representation="d0e1")])
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

doc_1 = Document(doc_id="1", elements=[Element(text_representation="d1e0"), Element(text_representation="d1e1")])
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

doc_2 = Document(doc_id="2", elements=[Element(text_representation="d2e0"), Element(text_representation="d2e1")])
doc_2.properties["_schema_temp"] = [
    {
        "name": "company_name",
        "type": {
            "type": "string",
            "description": "Name of the company",
            "examples": ["Gamma Inc"],
        },
    },
    {
        "name": "location",
        "type": {
            "type": "string",
            "description": "Company location",
            "examples": ["New York"],
        },
    },
]

doc_3 = Document(doc_id="3", elements=[Element(text_representation="d3e0"), Element(text_representation="d3e1")])
doc_3.properties["_schema_temp"] = [
    {
        "name": "company_name",
        "type": {
            "type": "string",
            "description": "Name of the company",
            "examples": ["Delta Co"],
        },
    },
    {
        "name": "ceo",
        "type": {
            "type": "string",
            "description": "CEO name",
            "examples": ["John Smith"],
        },
    },
]


class TestSchemaExtract:
    def test_intersection(self):
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
        context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
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

    def test_union(self):
        agg_schema_true = SchemaV2(
            properties=[
                {
                    "name": "company_name",
                    "type": {
                        "type": "string",
                        "description": "Name of the company",
                        "examples": ["Beta LLC", "Acme Corp"],
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
                {
                    "name": "revenue",
                    "type": {
                        "type": "float",
                        "description": "Annual revenue",
                        "examples": [1000000.0],
                    },
                },
            ]
        )

        docs = [doc_0, doc_1]
        context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
        read_ds = context.read.document(docs)
        schema_ext = SchemaExtract(
            read_ds.plan,
            step_through_strategy=BatchElements(batch_size=50),
            llm=FakeLLM(),
            prompt=FakeExtractionPrompt(),
        )
        ds = DocSet(context, schema_ext).reduce(union_of_fields)
        agg_schema_pred = ds.take()[0].properties.get("_schema", SchemaV2(properties=[]))

        assert len(agg_schema_pred.properties) == len(
            agg_schema_true.properties
        ), f"Expected {len(agg_schema_true.properties)} properties, got {len(agg_schema_pred.properties)}"
        assert set(p.name for p in agg_schema_pred.properties) == set(
            p.name for p in agg_schema_true.properties
        ), f"Expected property names {set(p.name for p in agg_schema_true.properties)}, got {set(p.name for p in agg_schema_pred.properties)}"
        for true_prop in agg_schema_true.properties:
            pred_prop = next((p for p in agg_schema_pred.properties if p.name == true_prop.name), None)
            assert pred_prop is not None, f"Property {true_prop.name} not found in predicted schema"
            print(f"Checking property {true_prop.name}")
            print(f"True property: {true_prop}")
            print(f"Predicted property: {pred_prop}")
            assert (
                pred_prop.type.type.value == true_prop.type.type.value
            ), f"Expected type {true_prop.type.type.value} for property {true_prop.name}, got {pred_prop.type.type.value}"
            assert (
                pred_prop.type.description == true_prop.type.description
            ), f"Expected description {true_prop.type.description} for property {true_prop.name}, got {pred_prop.type.description}"
            assert set(pred_prop.type.examples) == set(
                true_prop.type.examples
            ), f"Expected examples {true_prop.type.examples} for property {true_prop.name}, got {pred_prop.type.examples}"

    def test_frequency_filter(self):
        agg_schema_true = SchemaV2(
            properties=[
                {
                    "name": "company_name",
                    "type": {
                        "type": "string",
                        "description": "Name of the company",
                        "examples": ["Beta LLC", "Acme Corp", "Gamma Inc", "Delta Co"],
                    },
                },
                {
                    "name": "ceo",
                    "type": {
                        "type": "string",
                        "description": "CEO name",
                        "examples": ["Jane Doe", "John Smith"],
                    },
                },
            ]
        )

        docs = [doc_0, doc_1, doc_2, doc_3]
        context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
        read_ds = context.read.document(docs)
        schema_ext = SchemaExtract(
            read_ds.plan,
            step_through_strategy=BatchElements(batch_size=50),
            llm=FakeLLM(),
            prompt=FakeExtractionPrompt(),
        )
        ds = DocSet(context, schema_ext).reduce(make_freq_filter_fn(min_occurence_ratio=0.5))
        agg_schema_pred = ds.take()[0].properties.get("_schema", SchemaV2(properties=[]))

        assert len(agg_schema_pred.properties) == len(
            agg_schema_true.properties
        ), f"Expected {len(agg_schema_true.properties)} properties, got {len(agg_schema_pred.properties)}"
        assert set(p.name for p in agg_schema_pred.properties) == set(
            p.name for p in agg_schema_true.properties
        ), f"Expected property names {set(p.name for p in agg_schema_true.properties)}, got {set(p.name for p in agg_schema_pred.properties)}"
        for true_prop in agg_schema_true.properties:
            pred_prop = next((p for p in agg_schema_pred.properties if p.name == true_prop.name), None)
            assert pred_prop is not None, f"Property {true_prop.name} not found in predicted schema"
            print(f"Checking property {true_prop.name}")
            print(f"True property: {true_prop}")
            print(f"Predicted property: {pred_prop}")
            assert (
                pred_prop.type.type.value == true_prop.type.type.value
            ), f"Expected type {true_prop.type.type.value} for property {true_prop.name}, got {pred_prop.type.type.value}"
            assert (
                pred_prop.type.description == true_prop.type.description
            ), f"Expected description {true_prop.type.description} for property {true_prop.name}, got {pred_prop.type.description}"
            assert set(pred_prop.type.examples) == set(
                true_prop.type.examples
            ), f"Expected examples {true_prop.type.examples} for property {true_prop.name}, got {pred_prop.type.examples}"

    def test_schema_extract_empty_existing_schema(self):
        existing_schema = SchemaV2(properties=[])

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

        agg_schema_true = SchemaV2(
            properties=[
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
        )

        docs = [doc_0]
        context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
        read_ds = context.read.document(docs)
        schema_ext = SchemaExtract(
            read_ds.plan,
            step_through_strategy=BatchElements(batch_size=50),
            llm=FakeLLM(),
            prompt=FakeExtractionPrompt(),
            existing_schema=existing_schema,
        )
        ds = DocSet(context, schema_ext)
        agg_schema_pred = ds.take()[0].properties.get("_schema", SchemaV2(properties=[]))
        if existing_schema is not None and len(existing_schema.properties) > 0:
            for named_prop in existing_schema.properties:
                agg_schema_pred.properties.append(named_prop)

        assert len(agg_schema_pred.properties) == len(
            agg_schema_true.properties
        ), f"Expected {len(agg_schema_true.properties)} properties, got {len(agg_schema_pred.properties)}"
        assert set(p.name for p in agg_schema_pred.properties) == set(
            p.name for p in agg_schema_true.properties
        ), f"Expected property names {set(p.name for p in agg_schema_true.properties)}, got {set(p.name for p in agg_schema_pred.properties)}"
        for true_prop in agg_schema_true.properties:
            pred_prop = next((p for p in agg_schema_pred.properties if p.name == true_prop.name), None)
            assert pred_prop is not None, f"Property {true_prop.name} not found in predicted schema"
            print(f"Checking property {true_prop.name}")
            print(f"True property: {true_prop}")
            print(f"Predicted property: {pred_prop}")
            assert (
                pred_prop.type.type.value == true_prop.type.type.value
            ), f"Expected type {true_prop.type.type.value} for property {true_prop.name}, got {pred_prop.type.type.value}"
            assert (
                pred_prop.type.description == true_prop.type.description
            ), f"Expected description {true_prop.type.description} for property {true_prop.name}, got {pred_prop.type.description}"
            assert set(pred_prop.type.examples) == set(
                true_prop.type.examples
            ), f"Expected examples {true_prop.type.examples} for property {true_prop.name}, got {pred_prop.type.examples}"

    def test_schema_extract_existing_schema(self):
        existing_schema = SchemaV2(
            properties=[
                {
                    "name": "company_name",
                    "type": {
                        "type": "string",
                        "description": "Name of the company",
                        "examples": ["Acme Corp"],
                    },
                }
            ]
        )

        doc_0 = Document(
            doc_id="0", elements=[Element(text_representation="d0e0"), Element(text_representation="d0e1")]
        )
        doc_0.properties["_schema_temp"] = [
            {
                "name": "ceo",
                "type": {
                    "type": "string",
                    "description": "CEO name",
                    "examples": ["Jane Doe"],
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
        )

        docs = [doc_0]
        context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
        read_ds = context.read.document(docs)
        schema_ext = SchemaExtract(
            read_ds.plan,
            step_through_strategy=BatchElements(batch_size=50),
            llm=FakeLLM(),
            prompt=FakeExtractionPrompt(),
            existing_schema=existing_schema,
        )
        ds = DocSet(context, schema_ext)
        agg_schema_pred = ds.take()[0].properties.get("_schema", SchemaV2(properties=[]))
        if existing_schema is not None and len(existing_schema.properties) > 0:
            for named_prop in existing_schema.properties:
                agg_schema_pred.properties.append(named_prop)

        assert len(agg_schema_pred.properties) == len(
            agg_schema_true.properties
        ), f"Expected {len(agg_schema_true.properties)} properties, got {len(agg_schema_pred.properties)}"
        assert set(p.name for p in agg_schema_pred.properties) == set(
            p.name for p in agg_schema_true.properties
        ), f"Expected property names {set(p.name for p in agg_schema_true.properties)}, got {set(p.name for p in agg_schema_pred.properties)}"
        for true_prop in agg_schema_true.properties:
            pred_prop = next((p for p in agg_schema_pred.properties if p.name == true_prop.name), None)
            assert pred_prop is not None, f"Property {true_prop.name} not found in predicted schema"
            print(f"Checking property {true_prop.name}")
            print(f"True property: {true_prop}")
            print(f"Predicted property: {pred_prop}")
            assert (
                pred_prop.type.type.value == true_prop.type.type.value
            ), f"Expected type {true_prop.type.type.value} for property {true_prop.name}, got {pred_prop.type.type.value}"
            assert (
                pred_prop.type.description == true_prop.type.description
            ), f"Expected description {true_prop.type.description} for property {true_prop.name}, got {pred_prop.type.description}"
            assert set(pred_prop.type.examples) == set(
                true_prop.type.examples
            ), f"Expected examples {true_prop.type.examples} for property {true_prop.name}, got {pred_prop.type.examples}"

    def test_schema_extract_retry(self):
        docs = [doc_0, doc_1, doc_2, doc_3]
        context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
        read_ds = context.read.document(docs)
        schema_ext = SchemaExtract(
            read_ds.plan,
            step_through_strategy=BatchElements(batch_size=50),
            llm=FlakyLLM(max_fails=-1),
            prompt=FakeExtractionPrompt(),
        )
        ds = DocSet(context, schema_ext).reduce(make_freq_filter_fn(min_occurence_ratio=0.5))
        agg_schema_pred = ds.take()[0].properties.get("_schema", SchemaV2(properties=[]))

        assert len(agg_schema_pred.properties) == 0, f"Expected no properties, got {agg_schema_pred.properties}"

    def test_schema_extract_retry_success(self):
        agg_schema_true = SchemaV2(
            properties=[
                {
                    "name": "company_name",
                    "type": {
                        "type": "string",
                        "description": "Name of the company",
                        "examples": ["Beta LLC", "Acme Corp", "Gamma Inc", "Delta Co"],
                    },
                },
                {
                    "name": "ceo",
                    "type": {
                        "type": "string",
                        "description": "CEO name",
                        "examples": ["Jane Doe", "John Smith"],
                    },
                },
            ]
        )

        docs = [doc_0, doc_1, doc_2, doc_3]
        context = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
        read_ds = context.read.document(docs)
        schema_ext = SchemaExtract(
            read_ds.plan,
            step_through_strategy=BatchElements(batch_size=50),
            llm=FlakyLLM(max_fails=1),
            prompt=FakeExtractionPrompt(),
        )
        ds = DocSet(context, schema_ext).reduce(make_freq_filter_fn(min_occurence_ratio=0.5))
        agg_schema_pred = ds.take()[0].properties.get("_schema", SchemaV2(properties=[]))

        assert len(agg_schema_pred.properties) == len(
            agg_schema_true.properties
        ), f"Expected {len(agg_schema_true.properties)} properties, got {len(agg_schema_pred.properties)}"
        assert set(p.name for p in agg_schema_pred.properties) == set(
            p.name for p in agg_schema_true.properties
        ), f"Expected property names {set(p.name for p in agg_schema_true.properties)}, got {set(p.name for p in agg_schema_pred.properties)}"
        for true_prop in agg_schema_true.properties:
            pred_prop = next((p for p in agg_schema_pred.properties if p.name == true_prop.name), None)
            assert pred_prop is not None, f"Property {true_prop.name} not found in predicted schema"
            print(f"Checking property {true_prop.name}")
            print(f"True property: {true_prop}")
            print(f"Predicted property: {pred_prop}")
            assert (
                pred_prop.type.type.value == true_prop.type.type.value
            ), f"Expected type {true_prop.type.type.value} for property {true_prop.name}, got {pred_prop.type.type.value}"
            assert (
                pred_prop.type.description == true_prop.type.description
            ), f"Expected description {true_prop.type.description} for property {true_prop.name}, got {pred_prop.type.description}"
            assert set(pred_prop.type.examples) == set(
                true_prop.type.examples
            ), f"Expected examples {true_prop.type.examples} for property {true_prop.name}, got {pred_prop.type.examples}"
