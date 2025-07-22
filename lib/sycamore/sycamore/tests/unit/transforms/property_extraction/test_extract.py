import random
from typing import Optional

from sycamore.data.document import MetadataDocument, Document
from sycamore.data.element import Element
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import SycamorePrompt, RenderedPrompt, RenderedMessage
from sycamore.transforms.property_extraction.extract import Extract, _run_coros_threadsafe
from sycamore.transforms.property_extraction.strategy import NoSchemaSplitting, OneElementAtATime
from sycamore.utils.thread_local import ThreadLocal, ThreadLocalAccess, ADD_METADATA_TO_OUTPUT
from sycamore.schema import Schema, SchemaField


def test_run_coros():
    async def sometimes_recurse(n: int, tries: int = 0) -> int:
        if random.random() < 0.5:
            ThreadLocalAccess(ADD_METADATA_TO_OUTPUT).get().append(MetadataDocument(tries=tries))
            return n
        else:
            return await sometimes_recurse(n, tries + 1)

    nums = list(range(10))
    coros = [sometimes_recurse(n) for n in nums]
    meta = []
    with ThreadLocal(ADD_METADATA_TO_OUTPUT, meta):
        results = _run_coros_threadsafe(coros)

    assert results == nums
    assert len(meta) == 10


class FakeExtractionPrompt(SycamorePrompt):
    def render_multiple_elements(self, elts: list[Element], doc: Document) -> RenderedPrompt:
        return RenderedPrompt(
            messages=[
                RenderedMessage(role="user", content=f"docid={doc.doc_id}"),
                RenderedMessage(role="user", content=f"nelts={len(elts)}"),
                RenderedMessage(role="user", content=f"telts={len(doc.elements)}"),
            ]
        )


class FakeLLM(LLM):
    def __init__(self):
        super().__init__(model_name="fake", default_mode=LLMMode.ASYNC)

    def is_chat_mode(self):
        return True

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        return f"""{{
            "doc_id": "{prompt.messages[0].content[6:]}",
            "nelts": {prompt.messages[1].content[6:]},
            "telts": {prompt.messages[2].content[6:]}
        }}
        """

    async def generate_async(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        return self.generate(prompt=prompt, llm_kwargs=llm_kwargs)


class TestExtract:
    def test_extract(self):
        docs = [
            Document(
                doc_id="0",
                elements=[
                    Element(text_representation="d0e0"),
                    Element(text_representation="d0e1"),
                    Element(text_representation="d0e2"),
                ],
            ),
            Document(
                doc_id="1",
                elements=[
                    Element(text_representation="d1e0"),
                    Element(text_representation="d1e1"),
                ],
            ),
        ]

        schema = Schema(
            fields=[
                SchemaField(name="doc_id", field_type="str"),
                SchemaField(name="missing", field_type="str", default="Missing"),
                SchemaField(name="telts", field_type="int"),
            ]
        )

        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=FakeLLM(),
            prompt=FakeExtractionPrompt(),
        )

        extracted = extract.run(docs)
        assert extracted[0].field_to_value("properties.entity.doc_id") == docs[0].doc_id
        assert extracted[0].field_to_value("properties.entity.telts") == 3
        assert extracted[0].field_to_value("properties.entity.missing") == "Missing"

        assert extracted[1].field_to_value("properties.entity.doc_id") == docs[1].doc_id
        assert extracted[1].field_to_value("properties.entity.telts") == 2
        assert extracted[1].field_to_value("properties.entity.missing") == "Missing"
