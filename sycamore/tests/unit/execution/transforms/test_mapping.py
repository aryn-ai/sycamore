from typing import List

import pytest
import ray.data
from ray.data import ActorPoolStrategy

from sycamore.data import Document
from sycamore.execution import Node
from sycamore.execution.transforms import Map, FlatMap, MapBatch
from sycamore.execution.transforms.mapping import generate_map_batch_class_from_callable, Filter


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
    def test_map_function(self, mocker, function):
        node = mocker.Mock(spec=Node)
        mapping = Map(node, f=function)
        input_dataset = ray.data.from_items(
            [
                {"index": 1, "doc": "Members of a strike at Yale University."},
                {"index": 2, "doc": "A woman is speaking at a podium outdoors."},
            ]
        )
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = mapping.execute()
        dicts = output_dataset.take()
        assert dicts[0]["index"] == 2 and dicts[1]["index"] == 3

    class FlatMapClass:
        def __call__(self, doc: Document) -> List[Document]:
            return [doc, doc]

    @pytest.mark.parametrize("function", [flat_map_func, FlatMapClass])
    def test_flat_map(self, mocker, function):
        node = mocker.Mock(spec=Node)
        mapping = FlatMap(node, f=function)
        input_dataset = ray.data.from_items(
            [
                {"index": 1, "doc": "Members of a strike at Yale University."},
                {"index": 2, "doc": "A woman is speaking at a podium outdoors."},
            ]
        )
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
    def test_map_batch(self, mocker, function):
        node = mocker.Mock(spec=Node)
        mapping = MapBatch(node, f=function)
        input_dataset = ray.data.from_items(
            [
                {"index": 1, "doc": "Members of a strike at Yale University."},
                {"index": 2, "doc": "A woman is speaking at a podium outdoors."},
            ]
        )
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = mapping.execute()
        dicts = output_dataset.take()
        assert dicts[0]["index"] == 2 and dicts[1]["index"] == 3

    def test_generate_map_batch_class_from_callable(self):
        class BatchClass:
            def __call__(self, docs: List[Document]) -> List[Document]:
                for doc in docs:
                    doc["index"] += 1
                return docs

        ray_callable = generate_map_batch_class_from_callable(BatchClass())
        input_dataset = ray.data.from_items(
            [
                {"index": 1, "doc": "Members of a strike at Yale University."},
                {"index": 2, "doc": "A woman is speaking at a podium outdoors."},
            ]
        )
        output = input_dataset.map_batches(ray_callable, compute=ActorPoolStrategy())
        dicts = output.take()
        assert dicts[0]["index"] == 2 and dicts[1]["index"] == 3

    def test_generate_map_batch_filter_function(self, mocker):
        node = mocker.Mock(spec=Node)
        filtered = Filter(node, f=filter_func)
        input_dataset = ray.data.from_items(
            [
                {"index": 1, "doc": "Members of a strike at Yale University.", "properties": {"page_number": 1}},
                {"index": 2, "doc": "A woman is speaking at a podium outdoors.", "properties": {"page_number": 2}},
            ]
        )
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output = filtered.execute()

        dicts = output.take()
        assert len(dicts) == 1
        assert dicts[0]["properties"]["page_number"] == 1
