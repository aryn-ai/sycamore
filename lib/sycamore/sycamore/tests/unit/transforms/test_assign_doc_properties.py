import pytest
import ray.data

from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms.assign_doc_properties import AssignDocProperties
import unittest
from typing import List

class TestAssignDocProperties(unittest.TestCase):
    def setUp(self):
        self.input = Document(
            {
                "doc_id": "doc_id",
                "type": "pdf",
                "content": {"binary": None, "text": "text"},
                "parent_id": None,
                "properties": {"path": "s3://path"},
                "elements": [
                    {
                        "type": "title",
                        "content": {"binary": None, "text": "text1"},
                        "properties": {"property1": '{"test1": "from element1"}', "page_number": 1},
                    },
                    {
                        "type": "title",
                        "content": {"binary": None, "text": "text2"},
                        "properties": {'property1': '{"test1": "from element2"}', "page_number": 2},
                    },
                ],
            }
        )

    def test_assign_doc_propoerties(self):
        output = AssignDocProperties(None, ["title", "property1"]).run(self.input)
        print(output)
        assert 'entity' in output.get('properties').keys()
        assert output.get('properties').get('entity') is not None
        assert output.get('properties').get('entity')["test1"] == 'from element1'