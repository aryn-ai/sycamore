import functools
import json
import os

import sycamore
from sycamore import ExecMode, MaterializeSourceMode, DocSet
from sycamore.data import Document
from sycamore.datatype import DataType
from sycamore.llms import OpenAIModels
from sycamore.llms.anthropic import Anthropic
from sycamore.llms.config import AnthropicModels, GeminiModels
from sycamore.llms.gemini import Gemini
from sycamore.llms.openai import OpenAI
from sycamore.schema import SchemaV2
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import ArynPartitioner
from sycamore.transforms.property_extraction.attribution import LLMAttributionStrategy
from sycamore.transforms.property_extraction.extract import Extract
from sycamore.transforms.property_extraction.prompts import default_attribution_prompt
from sycamore.transforms.property_extraction.strategy import NPagesAtATime, NoSchemaSplitting
from sycamore.transforms.property_extraction.types import RichProperty

# from sycamore.utils.cache import DiskCache
from sycamore.utils.zip_traverse import zip_traverse


def test_take_first_boolean():
    TEST_DATA_DIR = TEST_DIR / "resources/data"
    aryn_api_key = os.getenv("ARYN_API_KEY")
    ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
    files = ["NVDA_10k_2025.pdf"]
    pick = files[0]
    pdf_path = TEST_DATA_DIR / f"pdfs/{pick}"
    ds = ctx.read.binary([str(pdf_path)], binary_format="pdf")
    ds = ds.partition(ArynPartitioner(aryn_api_key=aryn_api_key, text_mode="auto", table_mode="standard"))
    source_mode = MaterializeSourceMode.USE_STORED
    ds = ds.materialize(TEST_DATA_DIR / f"materialize/partitioned-{pick}", source_mode=source_mode)

    schema_file = "10k_schema_reduced.json"
    with open(TEST_DATA_DIR / schema_file) as f:
        j = json.load(f)

    pages_at_a_time = 5
    schema = SchemaV2.model_validate(j)
    api_key = os.getenv("OPENAI_API_KEY")
    cache = None
    llms = []
    llms.append(OpenAI(model_name=OpenAIModels.GPT_5_2, api_key=api_key, cache=cache))
    llms.append(Anthropic(model_name=AnthropicModels.CLAUDE_4_5_SONNET))
    llms.append(Gemini(model_name=GeminiModels.GEMINI_3_PRO_PREVIEW, api_key=os.getenv("GEMINI_API_KEY")))
    llms.append(Gemini(model_name=GeminiModels.GEMINI_2_5_PRO, api_key=os.getenv("GEMINI_API_KEY")))
    llm = llms[0]
    prompt = default_attribution_prompt
    extract = Extract(
        ds.plan,
        schema=schema,
        step_through_strategy=NPagesAtATime(pages_at_a_time),
        schema_partition_strategy=NoSchemaSplitting(),
        llm=llm,
        prompt=prompt,
        attribution_strategy=LLMAttributionStrategy(),
    )

    ds = DocSet(ctx, extract)
    # ds = ds.materialize(TEST_DATA_DIR / "materialize/extracted", source_mode=MaterializeSourceMode.RECOMPUTE)

    expected = True

    def check_props(document: Document) -> Document:
        print(f"Path: {document.properties['path']}")
        em = document.properties.get("entity_metadata")
        assert em is not None
        rp = RichProperty(name="", type=DataType.OBJECT, value=em)

        for k, v, p in zip_traverse(rp):
            if k == "major_customers_disclosed":
                assert expected is v[0].value
        return document

    ds = ds.map(functools.partial(check_props))
    ds.execute()


def test_take_first_array():
    TEST_DATA_DIR = TEST_DIR / "resources/data"
    aryn_api_key = os.getenv("ARYN_API_KEY")
    ctx = sycamore.init(exec_mode=ExecMode.LOCAL)
    files = ["NVDA_10k_2025.pdf"]
    pick = files[0]
    pdf_path = TEST_DATA_DIR / f"pdfs/{pick}"
    ds = ctx.read.binary([str(pdf_path)], binary_format="pdf")
    ds = ds.partition(ArynPartitioner(aryn_api_key=aryn_api_key, text_mode="auto", table_mode="standard"))
    source_mode = MaterializeSourceMode.USE_STORED
    ds = ds.materialize(TEST_DATA_DIR / f"materialize/partitioned-{pick}", source_mode=source_mode)

    schema_file = "10k_schema.json"
    with open(TEST_DATA_DIR / schema_file) as f:
        j = json.load(f)

    pages_at_a_time = 5
    schema = SchemaV2.model_validate(j)
    api_key = os.getenv("OPENAI_API_KEY")
    cache = None
    llms = []
    llms.append(OpenAI(model_name=OpenAIModels.GPT_5_2, api_key=api_key, cache=cache))
    llms.append(Anthropic(model_name=AnthropicModels.CLAUDE_4_5_SONNET))
    llms.append(Gemini(model_name=GeminiModels.GEMINI_3_PRO_PREVIEW, api_key=os.getenv("GEMINI_API_KEY")))
    llms.append(Gemini(model_name=GeminiModels.GEMINI_2_5_PRO, api_key=os.getenv("GEMINI_API_KEY")))
    llm = llms[0]
    prompt = default_attribution_prompt
    extract = Extract(
        ds.plan,
        schema=schema,
        step_through_strategy=NPagesAtATime(pages_at_a_time),
        schema_partition_strategy=NoSchemaSplitting(),
        llm=llm,
        prompt=prompt,
        attribution_strategy=LLMAttributionStrategy(),
    )

    ds = DocSet(ctx, extract)
    # ds = ds.materialize(TEST_DATA_DIR / "materialize/extracted", source_mode=MaterializeSourceMode.RECOMPUTE)

    expected = ["NORTH_AMERICA", "EUROPE", "ASIA"]

    def check_props(document: Document) -> Document:
        print(f"Path: {document.properties['path']}")
        em = document.properties.get("entity_metadata")
        assert em is not None
        rp = RichProperty(name="", type=DataType.OBJECT, value=em)

        got = set()
        for k, v, p in zip_traverse(rp):
            if k == "geographic_segments":
                for e in v[0].value:
                    _v = e.value
                    assert _v in expected
                    assert _v not in got
                    got.add(_v)

        return document

    ds = ds.map(functools.partial(check_props))
    ds.execute()
