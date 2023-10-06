from ray.util.client import ray

from functions import CharacterTokenizer
from plan_nodes import Node
from transforms import Coalesce
from transforms.coalesce import BBoxCoalescer


class TestCoalesce:
    doc = {
        "doc_id": "doc_id",
        "type": "pdf",
        "content": {"binary": None, "text": "text"},
        "parent_id": None,
        "properties": {"path": "s3://path"},
        "embedding": {"binary": None, "text": None},
        "elements": {
            "array": [
                {
                    "type": "title",
                    "content": {"binary": None, "text": "text1"},
                    "properties": {"coordinates": [(1, 2)], "page_number": 1},
                },
                {
                    "type": "table",
                    "content": {"binary": None, "text": "text2"},
                    "properties": {"page_name": "name", "coordinates": [(1, 2)], "coordinate_system": "pixel"},
                },
            ]
        },
    }

    def test_coalesce(self, mocker):
        node = mocker.Mock(spec=Node)
        coalesce = Coalesce(node, coalescer=BBoxCoalescer(tokenizer=CharacterTokenizer(), max_tokens_per_element=1000))
        input_dataset = ray.data.from_items([self.doc])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = coalesce.execute()