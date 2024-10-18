from sycamore.data import Document
from sycamore.transforms.standardizer import USStateStandardizer, StandardizeProperty, DateTimeStandardizer
import unittest
from datetime import date, datetime


class TestStandardizer(unittest.TestCase):
    def setUp(self):
        self.input = Document(
            {
                "doc_id": "doc_id",
                "type": "pdf",
                "properties": {
                    "path": "s3://path",
                    "entity": {"location": "Mountain View, CA", "dateTime": "March 17, 2023, 14.25 Local"},
                },
            }
        )

    def test_datetime(self):
        output = StandardizeProperty(
            None, standardizer=DateTimeStandardizer, path=[["properties", "entity", "dateTime"]]
        ).run(self.input)

        assert "properties" in output.keys()
        assert "entity" in output.properties.keys()
        assert output.properties.get("entity")["dateTime"] == "March 17, 2023 14:25:00"
        assert output.properties.get("entity")["day"] == date(2023, 3, 17)

    def test_datetime_format(self):
        output = StandardizeProperty(
            None,
            standardizer=DateTimeStandardizer,
            path=[["properties", "entity", "dateTime"]],
            date_format="%Y-%m-%d %H:%M:%S",
        ).run(self.input)

        assert "properties" in output.keys()
        assert "entity" in output.properties.keys()
        assert output.properties.get("entity")["dateTime"] == "2023-03-17 14:25:00"
        assert output.properties.get("entity")["day"] == date(2023, 3, 17)

    def test_location(self):
        output = StandardizeProperty(
            None, standardizer=USStateStandardizer, path=[["properties", "entity", "location"]]
        ).run(self.input)

        assert "properties" in output.keys()
        assert "entity" in output.properties.keys()
        assert "location" in output.properties.get("entity").keys()
        assert output.properties.get("entity")["location"] == "Mountain View, California"

    def test_datetime_no_local(self):
        input_copy = self.input.copy()
        input_copy["properties"]["entity"]["dateTime"] = "March 17, 2023, 14.25"
        date_standardizer = DateTimeStandardizer()
        output = StandardizeProperty(
            None, standardizer=date_standardizer, path=[["properties", "entity", "dateTime"]]
        ).run(input_copy)

        assert output.properties["entity"]["dateTime"] == "March 17, 2023 14:25:00"
        assert output.properties["entity"]["day"] == date(2023, 3, 17)

    def test_nonexistent_datetime_key(self):
        date_standardizer = DateTimeStandardizer()
        with self.assertRaises(KeyError):
            StandardizeProperty(
                None, standardizer=date_standardizer, path=[["properties", "entity", "nonexistent"]]
            ).run(self.input)


class TestUSStateStandardizer(unittest.TestCase):

    def setUp(self):
        self.standardizer = USStateStandardizer()

    def test_replace_abbreviations(self):
        # Test with single abbreviation
        input_string = "I live in CA."
        expected_output = "I live in California."
        self.assertEqual(self.standardizer.fixer(input_string), expected_output)

        input_string = "I am a big can of CALCIUM and TXT files."
        expected_output = "I am a big can of CALCIUM and TXT files."
        self.assertEqual(self.standardizer.fixer(input_string), expected_output)

        # Test with multiple abbreviations
        input_string = "I have been to NY, CA, and TX."
        expected_output = "I have been to New York, California, and Texas."
        self.assertEqual(self.standardizer.fixer(input_string), expected_output)

        # Test with mixed case abbreviations
        input_string = "We went to FL and GA last summer."
        expected_output = "We went to Florida and Georgia last summer."
        self.assertEqual(self.standardizer.fixer(input_string), expected_output)

        # Test with no abbreviations
        input_string = "No abbreviations here."
        expected_output = "No abbreviations here."
        self.assertEqual(self.standardizer.fixer(input_string), expected_output)

    def test_standardize(self):

        # Test with a simple document
        doc = {"location": "I have been to CA."}
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

        raw_datetime = "wrongdate"
        with self.assertRaises(ValueError):
            self.standardizer.fixer(raw_datetime)

        raw_dateTime = "2023123-07-15 10.30.00 Local"
        with self.assertRaises(ValueError):
            self.standardizer.fixer(raw_datetime)

        raw_dateTime = "2023-07-15 10.30.00 Local"
        expected_output = datetime(2023, 7, 15, 10, 30)
        self.assertEqual(self.standardizer.fixer(raw_dateTime), expected_output)

        # Test with datetime without 'Local'
        raw_dateTime = "2023-07-15 10.30.00"
        expected_output = datetime(2023, 7, 15, 10, 30)
        self.assertEqual(self.standardizer.fixer(raw_dateTime), expected_output)

        # Test with different datetime format
        raw_dateTime = "15/07/2023 10.30.00"
        expected_output = datetime(2023, 7, 15, 10, 30)
        self.assertEqual(self.standardizer.fixer(raw_dateTime), expected_output)

    def test_standardize(self):
        # Test with a simple document.
        doc = {"event": {"dateTime": "2023-07-15 10.30.00 Local"}}
        key_path = ["event", "dateTime"]
        expected_output = {"event": {"dateTime": "July 15, 2023 10:30:00", "day": date(2023, 7, 15)}}
        self.assertEqual(self.standardizer.standardize(doc, key_path), expected_output)

        # Test with a different date_format.
        doc = {"event": {"dateTime": "2023-07-15 10.30.00 Local"}}
        key_path = ["event", "dateTime"]
        expected_output = {"event": {"dateTime": "07/15/2023 10:30:00", "day": date(2023, 7, 15)}}
        self.assertEqual(
            self.standardizer.standardize(doc, key_path, date_format="%m/%d/%Y %H:%M:%S"),
            expected_output,
        )

        # Test with a different source key.
        doc = {"event": {"myDateTime": "2023-07-15 10.30.00 Local"}}
        key_path = ["event", "myDateTime"]
        expected_output = {
            "event": {
                "myDateTime": "07/15/2023 10:30:00",
                "dateTime": datetime(2023, 7, 15, 10, 30, 00),
                "day": date(2023, 7, 15),
            }
        }
        self.assertEqual(
            self.standardizer.standardize(doc, key_path, date_format="%m/%d/%Y %H:%M:%S"),
            expected_output,
        )

        # Test with a different source key and time zone in input.
        # For this test we need to remove the tzinfo from the datetime objects, since they
        # may not be equal.
        doc = {"event": {"myDateTime": "2023-07-15 10.30.00PDT"}}
        key_path = ["event", "myDateTime"]
        expected_output = {
            "event": {
                "myDateTime": "07/15/2023 10:30:00 PDT",
                "dateTime": datetime(2023, 7, 15, 10, 30, 00).replace(tzinfo=None),
                "day": date(2023, 7, 15),
            }
        }
        result = self.standardizer.standardize(doc, key_path, date_format="%m/%d/%Y %H:%M:%S %Z")
        result["event"]["dateTime"] = result["event"]["dateTime"].replace(tzinfo=None)
        self.assertEqual(
            result,
            expected_output,
        )

        # Test with a different source key and setting add_dateTime=False.
        doc = {"event": {"myDateTime": "2023-07-15 10.30.00 Local"}}
        key_path = ["event", "myDateTime"]
        expected_output = {
            "event": {
                "myDateTime": "07/15/2023 10:30:00",
                "day": date(2023, 7, 15),
            }
        }
        self.assertEqual(
            self.standardizer.standardize(doc, key_path, date_format="%m/%d/%Y %H:%M:%S", add_dateTime=False),
            expected_output,
        )

        # Test with nested document
        doc = {"user": {"activity": {"dateTime": "2023-07-15 10.30.00 Local"}}}
        key_path = ["user", "activity", "dateTime"]
        expected_output = {"user": {"activity": {"dateTime": "July 15, 2023 10:30:00", "day": date(2023, 7, 15)}}}
        self.assertEqual(self.standardizer.standardize(doc, key_path), expected_output)

        # Test with non-existent key
        doc = {"user": {"activity": {"dateTime": "2023-07-15 10.30.00 Local"}}}
        key_path = ["user", "activity", "date"]
        with self.assertRaises(KeyError):
            self.standardizer.standardize(doc, key_path)

        # Test with deeply nested document
        doc = {"system": {"log": {"entry": {"dateTime": "2023-07-15 10.30.00 Local"}}}}
        key_path = ["system", "log", "entry", "dateTime"]
        expected_output = {
            "system": {"log": {"entry": {"dateTime": "July 15, 2023 10:30:00", "day": date(2023, 7, 15)}}}
        }
        self.assertEqual(self.standardizer.standardize(doc, key_path), expected_output)

class TestIgnoreErrors(unittest.TestCase):

    def test_ignore_errors_value_missing(self):
        # Test with a document that has no key:value.

        standardizer = DateTimeStandardizer()
        doc = {"event": {"dateTime": ""}}
        key_path = ["event", "dateTime"]
        expected_output = {"event": {"dateTime": ""}}
        self.assertEqual(ignore_errors(doc, standardizer, key_path), expected_output)

 def test_ignore_errors_key_missing(self):
        # Test with a document that has no key:value.

        standardizer = DateTimeStandardizer()
        doc = {"event": {"coolKey": ""}}
        key_path = ["nonExistentKey"]
        expected_output = {"event": {"coolKey": ""}}
        self.assertEqual(ignore_errors(doc, standardizer, key_path), expected_output)


