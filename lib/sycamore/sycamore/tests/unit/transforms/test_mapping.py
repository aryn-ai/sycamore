from typing import List

import pytest
import ray.data

from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms import Map, FlatMap, MapBatch


def map_func(doc: Document) -> Document:
    doc["index"] += 1
    return doc


def flat_map_func(doc: Document) -> List[Document]:
    return [doc, doc]


def map_batch_func(docs: List[Document]) -> List[Document]:
    for doc in docs:
        doc["index"] += 1
    return docs


def filter_func(doc: Document) -> bool:
    return doc.properties["page_number"] == 1


class TestMapping:
    class MapClass:
        def __call__(self, doc: Document) -> Document:
            doc["index"] += 1
            return doc

    @pytest.mark.parametrize("function", [map_func, MapClass])
    def test_map_function(self, mocker, function) -> None:
        node = mocker.Mock(spec=Node)
        mapping = Map(node, f=function)
        dicts = [
            {"index": 1, "doc": "Members of a strike at Yale University."},
            {"index": 2, "doc": "A woman is speaking at a podium outdoors."},
        ]
        in_docs = [Document(d) for d in dicts]
        out_docs = mapping._local_process(in_docs)
        assert len(out_docs) == 2
        assert out_docs[0].data["index"] == 2
        assert out_docs[1].data["index"] == 3

        input_dataset = ray.data.from_items([{"doc": d.serialize()} for d in in_docs])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = mapping.execute()
        dicts = [Document.from_row(doc).data for doc in output_dataset.take()]

        # ray does not guarantee order preserving
        def sort_key(d: dict) -> int:
            return d["index"]

        dicts.sort(key=sort_key)
        print(dicts)
        assert dicts[0]["index"] == 2
        assert dicts[1]["index"] == 3

    class FlatMapClass:
        def __call__(self, doc: Document) -> List[Document]:
            assert isinstance(doc, Document)
            return [doc, doc]

    @pytest.mark.parametrize("function", [flat_map_func, FlatMapClass])
    def test_flat_map(self, mocker, function) -> None:
        node = mocker.Mock(spec=Node)
        mapping = FlatMap(node, f=function)
        dicts = [
            {"index": 1, "doc": "Members of a strike at Yale University."},
            {"index": 2, "doc": "A woman is speaking at a podium outdoors."},
        ]
        in_docs = [Document(d) for d in dicts]
        out_docs = mapping._local_process(in_docs)
        assert len(out_docs) == 4

        input_dataset = ray.data.from_items([{"doc": d.serialize()} for d in in_docs])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = mapping.execute()
        dicts = output_dataset.take()
        assert len(dicts) == 4

    class MapBatchClass:
        def __call__(self, docs: List[Document]) -> List[Document]:
            for doc in docs:
                doc["index"] += 1
            return docs

    @pytest.mark.parametrize("function", [map_batch_func, MapBatchClass])
    def test_map_batch(self, mocker, function) -> None:
        node = mocker.Mock(spec=Node)
        mapping = MapBatch(node, f=function)
        dicts = [
            {"index": 1, "doc": "Members of a strike at Yale University."},
            {"index": 2, "doc": "A woman is speaking at a podium outdoors."},
        ]
        input_dataset = ray.data.from_items([{"doc": Document(dict).serialize()} for dict in dicts])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = mapping.execute()
        dicts = [Document.from_row(doc).data for doc in output_dataset.take()]
        assert dicts[0]["index"] == 2 and dicts[1]["index"] == 3

    def test_flat_map_conflict(self, mocker) -> None:
        def func(doc: Document) -> List[Document]:
            if doc.doc_id == 1:
                return [
                    Document({"doc_id": 11, "parent_id": 1, "properties": {"author": "author"}}),
                    Document({"doc_id": 12, "properties": {"title": "title"}}),
                ]
            else:
                return [
                    Document({"doc_id": 21, "binary_representation": bytes("abc", "utf-8")}),
                    Document({"doc_id": 22, "text_representation": "abc"}),
                ]

        node = mocker.Mock(spec=Node)
        mapping = FlatMap(node, f=func)
        dicts = [
            {"doc_id": 1, "doc": "Members of a strike at Yale University."},
            {"doc_id": 2, "doc": "A woman is speaking at a podium outdoors."},
        ]
        input_dataset = ray.data.from_items([{"doc": Document(dict).serialize()} for dict in dicts])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = mapping.execute()
        batch = output_dataset.take_batch()
        assert len(batch["doc"]) == 4
