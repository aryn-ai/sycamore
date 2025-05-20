import unittest
from unittest.mock import MagicMock
import io
from pyarrow.fs import FileInfo, FileType, LocalFileSystem

from sycamore.connectors.file.tsv_scan import TsvScan
from sycamore.data import Document


class TestTsvScan(unittest.TestCase):
    def setUp(self):
        self.mock_filesystem = MagicMock(spec=LocalFileSystem)

    def _get_mock_file_info(self, path="test.tsv", is_file=True, ext="tsv"):
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
        tsv_scan = TsvScan(paths=["dummy_path"])
        self.assertEqual(tsv_scan.format(), "tsv")

    def test_simple_tsv_read_execute(self):
        paths = ["test.tsv"]
        tsv_scan = TsvScan(paths=paths, filesystem=self.mock_filesystem)
        
        file_data = [{'path': 'test.tsv', 'bytes': b'col_a\tcol_b\nval1\tval2\nval3\tval4'}]
        
        processed_docs_serialized = tsv_scan._process_ray_file_bytes(file_data[0])
        
        self.assertEqual(len(processed_docs_serialized), 2)
        
        doc1_data = processed_docs_serialized[0]["doc"]
        self.assertTrue(doc1_data["doc_id"].startswith("tsv-"))
        self.assertEqual(doc1_data["type"], "tsv")
        self.assertIsNone(doc1_data["text_representation"]) 
        self.assertEqual(doc1_data["properties"]["col_a"], "val1")
        self.assertEqual(doc1_data["properties"]["col_b"], "val2")
        self.assertEqual(doc1_data["properties"]["path"], "test.tsv")

        doc2_data = processed_docs_serialized[1]["doc"]
        self.assertTrue(doc2_data["doc_id"].startswith("tsv-"))
        self.assertEqual(doc2_data["type"], "tsv")
        self.assertIsNone(doc2_data["text_representation"])
        self.assertEqual(doc2_data["properties"]["col_a"], "val3")
        self.assertEqual(doc2_data["properties"]["col_b"], "val4")
        self.assertEqual(doc2_data["properties"]["path"], "test.tsv")

    def test_tsv_read_with_options_execute(self):
        paths = ["test_options.tsv"]
        # TsvScan defaults to tab delimiter, so no need to pass tsv_reader_options for that.
        # We can test other csv.DictReader options if needed, e.g. quotechar.
        tsv_scan = TsvScan(
            paths=paths,
            filesystem=self.mock_filesystem,
            # tsv_reader_options={'quotechar': '"'}, # Example if testing other options
            document_body_field="content",
            property_fields=["id"]
        )
        
        file_data = [{'path': 'test_options.tsv', 'bytes': b'id\tname\tcontent\nnote\n1\titem1\tHello TSV\tNote1'}]
        processed_docs_serialized = tsv_scan._process_ray_file_bytes(file_data[0])
        
        self.assertEqual(len(processed_docs_serialized), 1)
        doc_data = processed_docs_serialized[0]["doc"]
        
        self.assertTrue(doc_data["doc_id"].startswith("tsv-"))
        self.assertEqual(doc_data["type"], "tsv")
        self.assertEqual(doc_data["text_representation"], "Hello TSV")
        self.assertEqual(doc_data["properties"]["id"], "1")
        self.assertNotIn("name", doc_data["properties"]) 
        self.assertNotIn("notes", doc_data["properties"]) # notes was not in property_fields
        self.assertEqual(doc_data["properties"]["path"], "test_options.tsv")


    def test_process_file_basic(self):
        tsv_content = b"header1\theader2\ttext_col\nr1c1\tr1c2\tThis is body 1\nr2c1\tr2c2\tThis is body 2"
        mock_fs = self._get_mock_filesystem_for_process_file(tsv_content)
        file_info = self._get_mock_file_info(path="data/my.tsv")

        tsv_scan = TsvScan(
            paths=["data/my.tsv"], 
            filesystem=mock_fs, 
            document_body_field="text_col"
        )
        
        documents = tsv_scan.process_file(file_info)
        
        self.assertEqual(len(documents), 2)
        
        doc1 = documents[0]
        self.assertTrue(doc1.doc_id.startswith("tsv-"))
        self.assertEqual(doc1.type, "tsv")
        self.assertEqual(doc1.text_representation, "This is body 1")
        self.assertEqual(doc1.properties["header1"], "r1c1")
        self.assertEqual(doc1.properties["header2"], "r1c2")
        self.assertNotIn("text_col", doc1.properties) 
        self.assertEqual(doc1.properties["path"], "data/my.tsv")
        
        doc2 = documents[1]
        self.assertTrue(doc2.doc_id.startswith("tsv-"))
        self.assertEqual(doc2.type, "tsv")
        self.assertEqual(doc2.text_representation, "This is body 2")
        self.assertEqual(doc2.properties["header1"], "r2c1")
        self.assertEqual(doc2.properties["header2"], "r2c2")
        self.assertEqual(doc2.properties["path"], "data/my.tsv")

    def test_process_file_no_body_field_all_properties(self):
        tsv_content = b"header1\theader2\ttext_col\nr1c1\tr1c2\tBody value 1"
        mock_fs = self._get_mock_filesystem_for_process_file(tsv_content)
        file_info = self._get_mock_file_info(path="another.tsv")

        tsv_scan = TsvScan(paths=["another.tsv"], filesystem=mock_fs, document_body_field=None)
        documents = tsv_scan.process_file(file_info)

        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNone(doc.text_representation)
        self.assertEqual(doc.properties["header1"], "r1c1")
        self.assertEqual(doc.properties["header2"], "r1c2")
        self.assertEqual(doc.properties["text_col"], "Body value 1") 
        self.assertEqual(doc.properties["path"], "another.tsv")

    def test_process_file_selected_properties(self):
        tsv_content = b"id\tname\tage\tcity\n1\tAlice\t30\tNew York"
        mock_fs = self._get_mock_filesystem_for_process_file(tsv_content)
        file_info = self._get_mock_file_info(path="selected.tsv")

        tsv_scan = TsvScan(
            paths=["selected.tsv"], 
            filesystem=mock_fs, 
            property_fields=["name", "city"]
        )
        documents = tsv_scan.process_file(file_info)

        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIsNone(doc.text_representation)
        self.assertEqual(doc.properties["name"], "Alice")
        self.assertEqual(doc.properties["city"], "New York")
        self.assertNotIn("id", doc.properties)
        self.assertNotIn("age", doc.properties)
        self.assertEqual(doc.properties["path"], "selected.tsv")

    def test_process_file_empty_content(self):
        tsv_content = b"" 
        mock_fs = self._get_mock_filesystem_for_process_file(tsv_content)
        file_info = self._get_mock_file_info(path="empty.tsv")
        tsv_scan = TsvScan(paths=["empty.tsv"], filesystem=mock_fs)
        documents = tsv_scan.process_file(file_info)
        self.assertEqual(len(documents), 0)

    def test_process_file_empty_content_with_header(self):
        tsv_content = b"header1\theader2\n" 
        mock_fs = self._get_mock_filesystem_for_process_file(tsv_content)
        file_info = self._get_mock_file_info(path="header_only.tsv")
        tsv_scan = TsvScan(paths=["header_only.tsv"], filesystem=mock_fs)
        documents = tsv_scan.process_file(file_info)
        self.assertEqual(len(documents), 0)

    def test_process_file_malformed_tsv(self):
        # Test for lines with differing numbers of tabs. csv.DictReader handles this by 
        # filling missing fields with None or, if more fields than headers, putting them in a list under None key.
        # For basic TSV, often quotes are not special unless specified in csv.reader options.
        tsv_content = b'name\tvalue\tdescription\nitem1\tval1\tdesc1\nitem2\tval2\nitem3\tval3\tdesc3\textra'
        mock_fs = self._get_mock_filesystem_for_process_file(tsv_content)
        file_info = self._get_mock_file_info(path="malformed.tsv")
        tsv_scan = TsvScan(paths=["malformed.tsv"], filesystem=mock_fs)
        
        documents = tsv_scan.process_file(file_info)
        self.assertEqual(len(documents), 3)
        self.assertEqual(documents[0].properties["name"], "item1")
        self.assertEqual(documents[0].properties["value"], "val1")
        self.assertEqual(documents[0].properties["description"], "desc1")
        
        self.assertEqual(documents[1].properties["name"], "item2")
        self.assertEqual(documents[1].properties["value"], "val2")
        self.assertIsNone(documents[1].properties.get("description")) # Missing field

        self.assertEqual(documents[2].properties["name"], "item3")
        self.assertEqual(documents[2].properties["value"], "val3")
        self.assertEqual(documents[2].properties["description"], "desc3")
        # Extra fields might be handled by DictReader under a None key or ignored based on Python version/csv lib specifics.
        # For this test, we assume they are accessible if DictReader includes them or are ignored.
        # If they are under a None key: self.assertIn(None, documents[2].properties) and self.assertEqual(documents[2].properties[None], ['extra'])

    def test_process_file_non_tsv_extension(self):
        mock_fs = self._get_mock_filesystem_for_process_file(b"colA\tvalA")
        file_info_txt = self._get_mock_file_info(path="test.txt", ext="txt")
        
        tsv_scan = TsvScan(paths=["test.txt"], filesystem=mock_fs)
        documents = tsv_scan.process_file(file_info_txt)
        self.assertEqual(len(documents), 0)

    def test_process_file_is_not_file(self):
        mock_fs = self._get_mock_filesystem_for_process_file(b"colA\tvalA")
        file_info_dir = self._get_mock_file_info(path="a_directory.tsv", is_file=False)
        
        tsv_scan = TsvScan(paths=["a_directory.tsv"], filesystem=mock_fs)
        documents = tsv_scan.process_file(file_info_dir)
        self.assertEqual(len(documents), 0)


if __name__ == "__main__":
    unittest.main()
