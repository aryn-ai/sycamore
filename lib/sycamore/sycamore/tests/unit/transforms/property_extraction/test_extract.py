import asyncio
import time
from typing import Optional
import pytest

from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.llms.config import LLMModel
from sycamore.llms.llms import LLM, LLMMode, FakeLLM
from sycamore.llms.prompts.prompts import SycamorePrompt, RenderedPrompt, RenderedMessage
from sycamore.transforms.property_extraction.extract import Extract, ParallelBatches
from sycamore.transforms.property_extraction.strategy import (
    NoSchemaSplitting,
    OneElementAtATime,
    NPagesAtATime,
    NPagesAtATimeWithOverlap,
)
from sycamore.schema import (
    IntProperty,
    NamedProperty,
    RegexValidator,
    SchemaV2,
    StringProperty,
    DataType,
    ObjectProperty,
)


class FakeExtractionPrompt(SycamorePrompt):
    def render_multiple_elements(self, elts: list[Element], doc: Document) -> RenderedPrompt:
        return RenderedPrompt(
            messages=[
                RenderedMessage(role="user", content=f"docid={doc.doc_id}"),
                RenderedMessage(role="user", content=f"nelts={len(elts)}"),
                RenderedMessage(role="user", content=f"telts={len(doc.elements)}"),
            ]
        )


class LocalFakeLLM(LLM):
    def __init__(self, thinking_time: int = 0):
        super().__init__(model_name="fake", default_mode=LLMMode.ASYNC)
        self.ncalls = 0
        self.thinking_time = thinking_time

    def is_chat_mode(self):
        return True

    async def generate_async(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        self.ncalls += 1
        if self.thinking_time > 0:
            await asyncio.sleep(self.thinking_time)

        return f"""{{
            "doc_id": "{prompt.messages[0].content[6:]}",
            "nelts": {prompt.messages[1].content[6:]},
            "telts": {prompt.messages[2].content[6:]}
        }}
        """

    def generate(
        self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None, model: Optional[LLMModel] = None
    ) -> str:
        return asyncio.run(self.generate_async(prompt=prompt, llm_kwargs=llm_kwargs))


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
            llm=LocalFakeLLM(),
            prompt=FakeExtractionPrompt(),
        )

        extracted = extract.run(docs)
        assert extracted[0].field_to_value("properties.entity.doc_id") == "flarglhavn"
        assert extracted[0].field_to_value("properties.entity_metadata.doc_id").value == "flarglhavn"
        assert extracted[0].field_to_value("properties.entity.telts") == 3
        assert extracted[0].field_to_value("properties.entity_metadata.telts").value == 3
        assert extracted[0].field_to_value("properties.entity.missing") is None
        assert extracted[0].field_to_value("properties.entity_metadata.missing").value is None

        assert extracted[1].field_to_value("properties.entity.doc_id") == docs[1].doc_id
        assert extracted[1].field_to_value("properties.entity_metadata.doc_id").value == docs[1].doc_id
        assert extracted[1].field_to_value("properties.entity.telts") == 2
        assert extracted[1].field_to_value("properties.entity_metadata.telts").value == 2
        assert extracted[1].field_to_value("properties.entity.missing") is None
        assert extracted[1].field_to_value("properties.entity_metadata.missing").value is None

    def test_extract_serial(self):
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

        time_per_batch = 1
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=LocalFakeLLM(thinking_time=time_per_batch),
            prompt=FakeExtractionPrompt(),
        )

        t0 = time.time()
        extract.run(docs)
        elapsed = time.time() - t0
        assert elapsed >= time_per_batch * 3  # 3 elements is the larger of the two documents.

    def test_extract_parallel(self):
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

        time_per_batch = 1
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=LocalFakeLLM(thinking_time=time_per_batch),
            prompt=FakeExtractionPrompt(),
            batch_processing_mode=ParallelBatches(),
        )

        t0 = time.time()
        extract.run(docs)
        elapsed = time.time() - t0
        assert elapsed <= time_per_batch * 1.2

    def test_extract_pages_parallel(self):
        docs = [
            Document(
                doc_id="0",
                elements=[
                    Element(text_representation="d0e0", properties={"_element_index": 4, "page_number": 1}),
                    Element(text_representation="d0e1", properties={"_element_index": 9, "page_number": 2}),
                    Element(text_representation="d0e2", properties={"_element_index": 19, "page_number": 3}),
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
                    Element(text_representation="d1e0", properties={"_element_index": 40, "page_number": 1}),
                    Element(text_representation="d1e1", properties={"_element_index": 41, "page_number": 2}),
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

        time_per_batch = 1
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=NPagesAtATime(5),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=LocalFakeLLM(thinking_time=time_per_batch),
            prompt=FakeExtractionPrompt(),
            batch_processing_mode=ParallelBatches(),
        )

        t0 = time.time()
        extract.run(docs)
        elapsed = time.time() - t0
        assert elapsed <= time_per_batch * 1.2

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
            llm=LocalFakeLLM(),
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
                llm=LocalFakeLLM(),
                prompt=NotImplPrompt(),
                output_pydantic_models=False,
            )

        Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=LocalFakeLLM(),
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

        llm = LocalFakeLLM()
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

    def test_extract_parallel_validators(self):
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

        llm = LocalFakeLLM()
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=llm,
            prompt=FakeExtractionPrompt(),
            batch_processing_mode=ParallelBatches(),
        )

        extracted = extract.run(docs)
        # Incoming properties are assumed to be valid unless they say otherwise
        assert extracted[0].field_to_value("properties.entity_metadata.doc_id").is_valid
        assert not extracted[1].field_to_value("properties.entity_metadata.doc_id").is_valid
        assert llm.ncalls == 3 + 3 + 3

    def test_extract_validator_no_retry_null(self):
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

        llm = LocalFakeLLM()
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=llm,
            prompt=FakeExtractionPrompt(),
        )
        extracted = extract.run(docs)

        assert llm.ncalls == 5
        assert extracted[0].field_to_value("properties.entity.missing") is None
        assert extracted[1].field_to_value("properties.entity.missing") is None

    def test_double_object(self):
        docs = [Document(doc_id="0", elements=[Element(text_representation="aaa", properties={"_element_index": 2})])]
        schema = SchemaV2(
            properties=[
                NamedProperty(
                    name="outer", type=ObjectProperty(properties=[NamedProperty(name="inner", type=StringProperty())])
                )
            ]
        )

        llm = FakeLLM(return_value='{"outer": {"inner": "value"}}')
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=OneElementAtATime(),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=llm,
            prompt=FakeExtractionPrompt(),
        )
        extracted = extract.run(docs)
        assert extracted[0].field_to_value("properties.entity.outer.inner") == "value"

    def test_extract_pages_with_overlap(self):
        doc = Document(
            doc_id="0",
            elements=[
                Element(text_representation="d0e0", properties={"_element_index": 1, "page_number": 1}),
                Element(text_representation="d0e1", properties={"_element_index": 2, "page_number": 2}),
                Element(text_representation="d0e2", properties={"_element_index": 3, "page_number": 3}),
                Element(text_representation="d0e2", properties={"_element_index": 4, "page_number": 4}),
                Element(text_representation="d0e2", properties={"_element_index": 5, "page_number": 5}),
                Element(text_representation="d0e2", properties={"_element_index": 6, "page_number": 6}),
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
        )

        schema = SchemaV2(
            properties=[
                NamedProperty(name="doc_id", type=StringProperty()),
                NamedProperty(name="missing", type=StringProperty()),
                NamedProperty(name="telts", type=IntProperty()),
            ]
        )

        llm = LocalFakeLLM()
        step_size = 2
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=NPagesAtATime(step_size),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=llm,
            prompt=FakeExtractionPrompt(),
        )

        extract.run([doc])

        assert llm.ncalls == len(doc.elements) // step_size

        llm = LocalFakeLLM()
        step_size = 2
        overlap = 1
        extract = Extract(
            None,
            schema=schema,
            step_through_strategy=NPagesAtATimeWithOverlap(step_size, overlap=overlap),
            schema_partition_strategy=NoSchemaSplitting(),
            llm=llm,
            prompt=FakeExtractionPrompt(),
        )

        extract.run([doc])
        assert llm.ncalls == len(doc.elements) - 1
