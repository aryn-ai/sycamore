import pytest
import ray.data

from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms.standardizer import LocationStandardizer, Standardize_property, DateTimeStandardizer
import unittest
from typing import List
from datetime import date 

class TestStandardizer(unittest.TestCase):
    def setUp(self):
        self.input = Document(
            {
                "doc_id": "doc_id",
                "type": "pdf",
                "properties": {"path": "s3://path",
                               "entity":{
                                   "location":"Mountain View, CA",
                                   "dateTime": "March 17, 2023, 14.25 Local"
                                   }
                               }
            }
        )

    def test_datetime(self):
        # loc_standardizer = LocationStandardizer()
        date_standardizer = DateTimeStandardizer()

        output = Standardize_property(None, standardizer= date_standardizer ).run(self.input)
        assert 'properties' in output.keys()
        assert 'entity' in output.properties.keys()
        assert output.properties.get('entity')['dateTime'] == self.input.properties.get('entity')['dateTime']
        assert output.properties.get('entity')['day'] == date(2023, 3, 17)
    
    def test_location(self):
        loc_standardizer = LocationStandardizer()
        output = Standardize_property(None, standardizer= loc_standardizer ).run(self.input)
        print(output.properties)
        assert 'properties' in output.keys()
        assert 'entity' in output.properties.keys()
        assert 'location' in output.properties.get('entity').keys()
        assert output.properties.get('entity')['location'] == "Mountain View, California"