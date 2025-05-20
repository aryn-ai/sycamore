import unittest
from unittest.mock import MagicMock
import io
from pyarrow.fs import FileInfo, FileType, LocalFileSystem

from sycamore.connectors.file.csv_scan import CsvScan
from sycamore.data import Document


class TestCsvScan(unittest.TestCase):
    def setUp(self):
        self.mock_filesystem = MagicMock(spec=LocalFileSystem)

    def _get_mock_file_info(self, path="test.csv", is_file=True, ext="csv"):
        file_info = FileInfo(path) # Use actual FileInfo
        file_info.base_name = path.split("/")[-1]
        file_info.extension = ext
        file_info.type = FileType.File if is_file else FileType.Directory
        # file_info.is_file attribute doesn't exist directly, use file_info.type
        return file_info

    def _get_mock_filesystem_for_process_file(self, file_content_bytes: bytes):
        mock_fs = MagicMock(spec=LocalFileSystem)
        mock_stream = io.BytesIO(file_content_bytes) # Use real BytesIO for stream
        mock_fs.open_input_stream.return_value = mock_stream
        return mock_fs

    def test_format_method(self):
        csv_scan = CsvScan(paths=["dummy_path"])
        self.assertEqual(csv_scan.format(), "csv")

    def test_simple_csv_read_execute(self):
        paths = ["test.csv"]
        csv_scan = CsvScan(paths=paths, filesystem=self.mock_filesystem)
        
        file_data = [{'path': 'test.csv', 'bytes': b'col_a,col_b\nval1,val2\nval3,val4'}]
        
        # Directly test _process_ray_file_bytes as it contains the core logic
        processed_docs_serialized = csv_scan._process_ray_file_bytes(file_data[0])
        
        self.assertEqual(len(processed_docs_serialized), 2)
        
        doc1_data = processed_docs_serialized[0]["doc"]
        self.assertTrue(doc1_data["doc_id"].startswith("csv-"))
        self.assertEqual(doc1_data["type"], "csv")
        self.assertIsNone(doc1_data["text_representation"]) # No document_body_field
        self.assertEqual(doc1_data["properties"]["col_a"], "val1")
        self.assertEqual(doc1_data["properties"]["col_b"], "val2")
        self.assertEqual(doc1_data["properties"]["path"], "test.csv")

        doc2_data = processed_docs_serialized[1]["doc"]
        self.assertTrue(doc2_data["doc_id"].startswith("csv-"))
        self.assertEqual(doc2_data["type"], "csv")
        self.assertIsNone(doc2_data["text_representation"])
        self.assertEqual(doc2_data["properties"]["col_a"], "val3")
        self.assertEqual(doc2_data["properties"]["col_b"], "val4")
        self.assertEqual(doc2_data["properties"]["path"], "test.csv")

    def test_csv_read_with_options_execute(self):
        paths = ["test_options.csv"]
        csv_scan = CsvScan(
            paths=paths,
            filesystem=self.mock_filesystem,
            csv_reader_options={'delimiter': ';'},
            document_body_field="content",
            property_fields=["id"]
        )
        
        file_data = [{'path': 'test_options.csv', 'bytes': b'id;name;content\n1;item1;this is content1'}]
        processed_docs_serialized = csv_scan._process_ray_file_bytes(file_data[0])
        
        self.assertEqual(len(processed_docs_serialized), 1)
        doc_data = processed_docs_serialized[0]["doc"]
        
        self.assertTrue(doc_data["doc_id"].startswith("csv-"))
        self.assertEqual(doc_data["type"], "csv")
        self.assertEqual(doc_data["text_representation"], "this is content1")
        self.assertEqual(doc_data["properties"]["id"], "1")
        self.assertNotIn("name", doc_data["properties"]) # name was not in property_fields
        self.assertEqual(doc_data["properties"]["path"], "test_options.csv")


    def test_process_file_basic(self):
        csv_content = b"header1,header2,text_col\nr1c1,r1c2,This is body 1\nr2c1,r2c2,This is body 2"
        mock_fs = self._get_mock_filesystem_for_process_file(csv_content)
        file_info = self._get_mock_file_info(path="data/my.csv")

        csv_scan = CsvScan(
            paths=["data/my.csv"], 
            filesystem=mock_fs, 
            document_body_field="text_col"
        )
        
        documents = csv_scan.process_file(file_info)
        
        self.assertEqual(len(documents), 2)
        
        doc1 = documents[0]
        self.assertTrue(doc1.doc_id.startswith("csv-"))
        self.assertEqual(doc1.type, "csv")
        self.assertEqual(doc1.text_representation, "This is body 1")
        self.assertEqual(doc1.properties["header1"], "r1c1")
        self.assertEqual(doc1.properties["header2"], "r1c2")
        self.assertNotIn("text_col", doc1.properties) # Should be body, not property
        self.assertEqual(doc1.properties["path"], "data/my.csv")
        
        doc2 = documents[1]
        self.assertTrue(doc2.doc_id.startswith("csv-"))
        self.assertEqual(doc2.type, "csv")
        self.assertEqual(doc2.text_representation, "This is body 2")
        self.assertEqual(doc2.properties["header1"], "r2c1")
        self.assertEqual(doc2.properties["header2"], "r2c2")
        self.assertEqual(doc2.properties["path"], "data/my.csv")

    def test_process_file_no_body_field_all_properties(self):
        csv_content = b"header1,header2,text_col\nr1c1,r1c2,Body value 1"
        mock_fs = self._get_mock_filesystem_for_process_file(csv_content)
        file_info = self._get_mock_file_info(path="another.csv")

        csv_scan = CsvScan(paths=["another.csv"], filesystem=mock_fs, document_body_field=None)
        documents = csv_scan.process_file(file_info)

        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNone(doc.text_representation)
        self.assertEqual(doc.properties["header1"], "r1c1")
        self.assertEqual(doc.properties["header2"], "r1c2")
        self.assertEqual(doc.properties["text_col"], "Body value 1") # Now it's a property
        self.assertEqual(doc.properties["path"], "another.csv")

    def test_process_file_selected_properties(self):
        csv_content = b"id,name,age,city\n1,Alice,30,New York"
        mock_fs = self._get_mock_filesystem_for_process_file(csv_content)
        file_info = self._get_mock_file_info(path="selected.csv")

        csv_scan = CsvScan(
            paths=["selected.csv"], 
            filesystem=mock_fs, 
            property_fields=["name", "city"]
        )
        documents = csv_scan.process_file(file_info)

        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNone(doc.text_representation)
        self.assertEqual(doc.properties["name"], "Alice")
        self.assertEqual(doc.properties["city"], "New York")
        self.assertNotIn("id", doc.properties)
        self.assertNotIn("age", doc.properties)
        self.assertEqual(doc.properties["path"], "selected.csv")

    def test_process_file_empty_content(self):
        csv_content = b"" # Empty file
        mock_fs = self._get_mock_filesystem_for_process_file(csv_content)
        file_info = self._get_mock_file_info(path="empty.csv")
        csv_scan = CsvScan(paths=["empty.csv"], filesystem=mock_fs)
        documents = csv_scan.process_file(file_info)
        self.assertEqual(len(documents), 0)

    def test_process_file_empty_content_with_header(self):
        csv_content = b"header1,header2\n" # Only header
        mock_fs = self._get_mock_filesystem_for_process_file(csv_content)
        file_info = self._get_mock_file_info(path="header_only.csv")
        csv_scan = CsvScan(paths=["header_only.csv"], filesystem=mock_fs)
        documents = csv_scan.process_file(file_info)
        self.assertEqual(len(documents), 0)

    def test_process_file_malformed_csv(self):
        # csv.DictReader is generally robust to missing fields in rows.
        # It will fill them with None if a field is missing.
        # If quotes are mismatched, it might raise an error or misinterpret.
        # This test assumes basic robustness of csv.DictReader.
        csv_content = b'name,value\nitem1,"description with a newline\nstill part of description"\nitem2,val2'
        mock_fs = self._get_mock_filesystem_for_process_file(csv_content)
        file_info = self._get_mock_file_info(path="malformed.csv")
        csv_scan = CsvScan(paths=["malformed.csv"], filesystem=mock_fs)
        
        # Depending on csv library's strictness, this might produce 2 docs or fewer if error.
        # Python's csv.DictReader usually handles embedded newlines in quoted fields.
        documents = csv_scan.process_file(file_info)
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].properties["name"], "item1")
        self.assertEqual(documents[0].properties["value"], "description with a newline\nstill part of description")
        self.assertEqual(documents[1].properties["name"], "item2")
        self.assertEqual(documents[1].properties["value"], "val2")

    def test_process_file_non_csv_extension(self):
        mock_fs = self._get_mock_filesystem_for_process_file(b"colA\nvalA")
        # Use a non-csv extension, or a path that process_file might filter out
        file_info_txt = self._get_mock_file_info(path="test.txt", ext="txt")
        
        csv_scan = CsvScan(paths=["test.txt"], filesystem=mock_fs)
        documents = csv_scan.process_file(file_info_txt)
        self.assertEqual(len(documents), 0)

    def test_process_file_is_not_file(self):
        mock_fs = self._get_mock_filesystem_for_process_file(b"colA\nvalA")
        file_info_dir = self._get_mock_file_info(path="a_directory.csv", is_file=False)
        
        csv_scan = CsvScan(paths=["a_directory.csv"], filesystem=mock_fs)
        documents = csv_scan.process_file(file_info_dir)
        self.assertEqual(len(documents), 0)


if __name__ == "__main__":
    unittest.main()
