from typing import Optional
import pytest

from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import SycamorePrompt, RenderedPrompt, RenderedMessage
from sycamore.transforms.property_extraction.extract import Extract
from sycamore.transforms.property_extraction.strategy import NoSchemaSplitting, OneElementAtATime
from sycamore.schema import IntProperty, NamedProperty, RegexValidator, SchemaV2, StringProperty, DataType


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
        self.ncalls = 0

    def is_chat_mode(self):
        return True

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        self.ncalls += 1
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
                    Element(text_representation="d0e0", properties={"_element_index": 4}),
                    Element(text_representation="d0e1", properties={"_element_index": 9}),
                    Element(text_representation="d0e2", properties={"_element_index": 19}),
                ],
                properties={
                    "entity": {"doc_id": "flarglhavn"},
                    "entity_metadata": {
                        "doc_id": {
                            "name": "doc_id",
                            "type": DataType.STRING,
                            "value": "flarglhavn",
                            "attribution": {"element_indices": [0], "page": 0, "bbox": [0, 0, 1, 1]},
                        }
                    },
                },
            ),
            Document(
                doc_id="1",
                elements=[
                    Element(text_representation="d1e0", properties={"_element_index": 40}),
                    Element(text_representation="d1e1", properties={"_element_index": 41}),
                ],
            ),
        ]

        schema = SchemaV2(
            properties=[
                NamedProperty(name="doc_id", type=StringProperty()),
                NamedProperty(name="missing", type=StringProperty()),
                NamedProperty(name="telts", type=IntProperty()),
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
        assert extracted[0].field_to_value("properties.entity.doc_id") == "flarglhavn"
        assert extracted[0].field_to_value("properties.entity_metadata.doc_id").value == "flarglhavn"
        assert extracted[0].field_to_value("properties.entity.telts") == 3
        assert extracted[0].field_to_value("properties.entity_metadata.telts").value == 3
        assert extracted[0].field_to_value("properties.entity.missing") is None
        assert extracted[0].field_to_value("properties.entity_metadata.missing") is None

        assert extracted[1].field_to_value("properties.entity.doc_id") == docs[1].doc_id
        assert extracted[1].field_to_value("properties.entity_metadata.doc_id").value == docs[1].doc_id
        assert extracted[1].field_to_value("properties.entity.telts") == 2
        assert extracted[1].field_to_value("properties.entity_metadata.telts").value == 2
        assert extracted[1].field_to_value("properties.entity.missing") is None
        assert extracted[1].field_to_value("properties.entity_metadata.missing") is None

    def test_extract_to_nonpydantic(self):
        docs = [
            Document(
                doc_id="0",
                elements=[
                    Element(text_representation="d0e0", properties={"_element_index": 4}),
                    Element(text_representation="d0e1", properties={"_element_index": 9}),
                    Element(text_representation="d0e2", properties={"_element_index": 19}),
                ],
                properties={
                    "entity": {"doc_id": "flarglhavn"},
                    "entity_metadata": {
                        "doc_id": {
                            "name": "doc_id",
                            "type": DataType.STRING,
                            "value": "flarglhavn",
                            "attribution": {"element_indices": [0], "page": 0, "bbox": [0, 0, 1, 1]},
                        }
                    },
                },
            ),
            Document(
                doc_id="1",
                elements=[
                    Element(text_representation="d1e0", properties={"_element_index": 40}),
                    Element(text_representation="d1e1", properties={"_element_index": 42}),
                ],
            ),
        ]

        schema = SchemaV2(
            properties=[
                NamedProperty(name="doc_id", type=StringProperty()),
                NamedProperty(name="missing", type=StringProperty()),
                NamedProperty(name="telts", type=IntProperty()),
            ]
        )

        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=FakeLLM(),
            prompt=FakeExtractionPrompt(),
            output_pydantic_models=False,
        )

        extracted = extract.run(docs)
        assert extracted[0].field_to_value("properties.entity.doc_id") == "flarglhavn"
        assert extracted[0].field_to_value("properties.entity_metadata.doc_id.value") == "flarglhavn"
        assert extracted[0].field_to_value("properties.entity.telts") == 3
        assert extracted[0].field_to_value("properties.entity_metadata.telts.value") == 3
        assert extracted[0].field_to_value("properties.entity.missing") is None
        assert extracted[0].field_to_value("properties.entity_metadata.missing.value") is None

        assert extracted[1].field_to_value("properties.entity.doc_id") == docs[1].doc_id
        assert extracted[1].field_to_value("properties.entity_metadata.doc_id.value") == docs[1].doc_id
        assert extracted[1].field_to_value("properties.entity.telts") == 2
        assert extracted[1].field_to_value("properties.entity_metadata.telts.value") == 2
        assert extracted[1].field_to_value("properties.entity.missing") is None
        assert extracted[1].field_to_value("properties.entity_metadata.missing.value") is None

    def test_extract_with_bad_prompts(self):
        class NotImplPrompt(SycamorePrompt):
            pass

        class ImplButCrashPrompt(SycamorePrompt):
            def render_multiple_elements(self, elts: list[Element], doc: Document) -> RenderedPrompt:
                1 / 0
                return RenderedPrompt(messages=[])

        schema = SchemaV2(
            properties=[
                NamedProperty(name="doc_id", type=StringProperty()),
                NamedProperty(name="missing", type=StringProperty()),
                NamedProperty(name="telts", type=IntProperty()),
            ]
        )

        with pytest.raises(NotImplementedError):
            Extract(
                None,
                schema=schema,
                step_through_strategy=OneElementAtATime(),
                schema_partition_strategy=NoSchemaSplitting(),
                llm=FakeLLM(),
                prompt=NotImplPrompt(),
                output_pydantic_models=False,
            )

        Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=FakeLLM(),
            prompt=ImplButCrashPrompt(),
            output_pydantic_models=False,
        )

    def test_extract_validators(self):
        docs = [
            Document(
                doc_id="0",
                elements=[
                    Element(text_representation="d0e0", properties={"_element_index": 4}),
                    Element(text_representation="d0e1", properties={"_element_index": 9}),
                    Element(text_representation="d0e2", properties={"_element_index": 19}),
                ],
                properties={
                    "entity": {"doc_id": "flarglhavn"},
                    "entity_metadata": {
                        "doc_id": {
                            "name": "doc_id",
                            "type": DataType.STRING,
                            "value": "flarglhavn",
                            "attribution": {"element_indices": [0], "page": 0, "bbox": [0, 0, 1, 1]},
                        }
                    },
                },
            ),
            Document(
                doc_id="1",
                elements=[
                    Element(text_representation="d1e0", properties={"_element_index": 40}),
                    Element(text_representation="d1e1", properties={"_element_index": 41}),
                ],
            ),
        ]

        schema = SchemaV2(
            properties=[
                NamedProperty(
                    name="doc_id",
                    type=StringProperty(validators=[RegexValidator(regex=r"regexthatdoesntmatch", n_retries=3)]),
                ),
                NamedProperty(name="telts", type=IntProperty()),
            ]
        )

        llm = FakeLLM()
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=llm,
            prompt=FakeExtractionPrompt(),
        )

        extracted = extract.run(docs)
        # Incoming properties are assumed to be valid unless they say otherwise
        assert extracted[0].field_to_value("properties.entity_metadata.doc_id").is_valid
        assert not extracted[1].field_to_value("properties.entity_metadata.doc_id").is_valid
        assert llm.ncalls == 1 + 3 + 3

    def test_extract_validator_retry_unpredictable_prop(self):
        docs = [
            Document(
                doc_id="0",
                elements=[
                    Element(text_representation="d0e0", properties={"_element_index": 4}),
                    Element(text_representation="d0e1", properties={"_element_index": 9}),
                    Element(text_representation="d0e2", properties={"_element_index": 19}),
                ],
            ),
            Document(
                doc_id="1",
                elements=[
                    Element(text_representation="d1e0", properties={"_element_index": 40}),
                    Element(text_representation="d1e1", properties={"_element_index": 41}),
                ],
            ),
        ]
        schema = SchemaV2(
            properties=[
                NamedProperty(name="missing", type=StringProperty()),
            ]
        )

        llm = FakeLLM()
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=llm,
            prompt=FakeExtractionPrompt(),
        )
        extracted = extract.run(docs)

        from sycamore.transforms.property_extraction.extract import MAX_RETRIES

        assert llm.ncalls == MAX_RETRIES * 5
        assert extracted[0].field_to_value("properties.entity.missing") is None
        assert extracted[1].field_to_value("properties.entity.missing") is None
