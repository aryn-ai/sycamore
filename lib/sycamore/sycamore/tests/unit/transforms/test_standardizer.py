import pytest
import ray.data

from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms.standardizer import LocationStandardizer, StandardizeProperty, DateTimeStandardizer
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
        date_standardizer = DateTimeStandardizer()

        output = StandardizeProperty(None, standardizer= date_standardizer, path=[["properties", "entity", "dateTime"]] ).run(self.input)
        assert 'properties' in output.keys()
        assert 'entity' in output.properties.keys()
        assert output.properties.get('entity')['dateTime'] == "March 17, 2023, 14:25 "
        assert output.properties.get('entity')['day'] == date(2023, 3, 17)
    
    def test_location(self):
        loc_standardizer = LocationStandardizer()
        output = StandardizeProperty(None, standardizer= loc_standardizer, path=[["properties", "entity", "location"]]  ).run(self.input)
        
        assert 'properties' in output.keys()
        assert 'entity' in output.properties.keys()
        assert 'location' in output.properties.get('entity').keys()
        assert output.properties.get('entity')['location'] == "Mountain View, California"


class TestLocationStandardizer(unittest.TestCase):

    def setUp(self):
        self.standardizer = LocationStandardizer()

    def test_replace_abbreviations(self):
        # Test with single abbreviation
        input_string = "I live in CA."
        expected_output = "I live in California."
        self.assertEqual(self.standardizer.replace_abbreviations(input_string), expected_output)

        # Test with multiple abbreviations
        input_string = "I have been to NY, CA, and TX."
        expected_output = "I have been to New York, California, and Texas."
        self.assertEqual(self.standardizer.replace_abbreviations(input_string), expected_output)

        # Test with mixed case abbreviations
        input_string = "We went to FL and GA last summer."
        expected_output = "We went to Florida and Georgia last summer."
        self.assertEqual(self.standardizer.replace_abbreviations(input_string), expected_output)

        # Test with no abbreviations
        input_string = "No abbreviations here."
        expected_output = "No abbreviations here."
        self.assertEqual(self.standardizer.replace_abbreviations(input_string), expected_output)

    def test_standardize(self):
        
        # Test with a simple document
        doc =  ({"location": "I have been to CA."})
        key_path = ["location"]
        expected_output = {"location": "I have been to California."}
        print(self.standardizer.standardize(doc, key_path))
        self.assertEqual(self.standardizer.standardize(doc, key_path), expected_output)

        # Test with nested document
        doc = {"address": {"state": "I have been to NY."}}
        key_path = ["address", "state"]
        expected_output = {"address": {"state": "I have been to New York."}}
        self.assertEqual(self.standardizer.standardize(doc, key_path), expected_output)

        # Test with non-existent key
        doc = {"address": {"state": "I have been to NY."}}
        key_path = ["address", "country"]
        with self.assertRaises(KeyError):
            self.standardizer.standardize(doc, key_path)

        # Test with deeply nested document
        doc = {"user": {"address": {"state": "My home is TX."}}}
        key_path = ["user", "address", "state"]
        expected_output = {"user": {"address": {"state": "My home is Texas."}}}
        self.assertEqual(self.standardizer.standardize(doc, key_path), expected_output)

class TestDateTimeStandardizer(unittest.TestCase):

    def setUp(self):
        self.standardizer = DateTimeStandardizer()

    def test_fix_date(self):
        # Test with typical datetime format
        raw_dateTime = "2023-07-15 10.30.00 Local"
        expected_output = ("2023-07-15 10:30:00 ", date(2023, 7, 15))
        self.assertEqual(self.standardizer.fix_date(raw_dateTime), expected_output)

        # Test with datetime without 'Local'
        raw_dateTime = "2023-07-15 10.30.00"
        expected_output = ("2023-07-15 10:30:00", date(2023, 7, 15))
        self.assertEqual(self.standardizer.fix_date(raw_dateTime), expected_output)

        # Test with different datetime format
        raw_dateTime = "15/07/2023 10.30.00"
        expected_output = ("15/07/2023 10:30:00", date(2023, 7, 15))
        self.assertEqual(self.standardizer.fix_date(raw_dateTime), expected_output)

    def test_standardize(self):
        # Test with a simple document
        doc = {"event": {"dateTime": "2023-07-15 10.30.00 Local"}}
        key_path = ["event", "dateTime"]
        expected_output = {"event": {"dateTime": "2023-07-15 10:30:00 ", "day": date(2023, 7, 15)}}
        self.assertEqual(self.standardizer.standardize(doc, key_path), expected_output)

        # Test with nested document
        doc = {"user": {"activity": {"dateTime": "2023-07-15 10.30.00 Local"}}}
        key_path = ["user", "activity", "dateTime"]
        expected_output = {"user": {"activity": {"dateTime": "2023-07-15 10:30:00 ", "day": date(2023, 7, 15)}}}
        self.assertEqual(self.standardizer.standardize(doc, key_path), expected_output)

        # Test with non-existent key
        doc = {"user": {"activity": {"dateTime": "2023-07-15 10.30.00 Local"}}}
        key_path = ["user", "activity", "date"]
        with self.assertRaises(KeyError):
            self.standardizer.standardize(doc, key_path)

        # Test with deeply nested document
        doc = {"system": {"log": {"entry": {"dateTime": "2023-07-15 10.30.00 Local"}}}}
        key_path = ["system", "log", "entry", "dateTime"]
        expected_output = {"system": {"log": {"entry": {"dateTime": "2023-07-15 10:30:00 ", "day": date(2023, 7, 15)}}}}
        self.assertEqual(self.standardizer.standardize(doc, key_path), expected_output)
