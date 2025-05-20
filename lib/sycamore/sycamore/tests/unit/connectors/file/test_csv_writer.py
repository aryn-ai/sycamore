import unittest
import csv
from unittest.mock import MagicMock, patch
import pyarrow.fs as pa_fs

# If _CsvBlockDataSink is intended to be private, this import might need adjustment
# For now, assume it's importable for direct testing as per instructions.
from sycamore.connectors.file.csv_writer import CsvWriter, _CsvBlockDataSink
from sycamore.connectors.file.file_writer_ray import generate_filename # Import for potential mocking
from sycamore.data import Document
from sycamore.plan_nodes import Node


class TestCsvWriter(unittest.TestCase):
    def setUp(self):
        self.fs = pa_fs.InMemoryFileSystem()
        self.output_path = "/test_output/"
        # Ensure the base path exists in the InMemoryFileSystem
        self.fs.create_dir(self.output_path)

    def _create_doc(self, doc_id: str, text_representation: str, properties: dict = None, type: str = "test") -> Document:
        return Document({
            "doc_id": doc_id,
            "type": type,
            "text_representation": text_representation,
            "properties": properties or {}
        })

    def tearDown(self):
        # Clear the InMemoryFileSystem if necessary, though it's usually fresh per instance
        # For InMemoryFileSystem, files are cleared when the instance is gone.
        # If using a persistent mock or a real temp directory, cleanup would be needed here.
        pass

    @patch("sycamore.connectors.file.csv_writer.generate_filename")
    def test_csv_writer_basic_write(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "part-000000.csv"
        
        docs = [
            self._create_doc("d1", "text1", {"name": "Alice", "age": 30}),
            self._create_doc("d2", "text2", {"name": "Bob", "age": 25}),
        ]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _CsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["doc_id", "name", "age", "text_representation"],
            csv_writer_options={'delimiter': ','},
            write_header=True,
        )
        
        written_path = sink._write_single_block(serialized_docs, 0)
        self.assertEqual(written_path, self.output_path + "part-000000.csv")

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = (
            "doc_id,name,age,text_representation\r\n"  # csv module uses \r\n by default
            "d1,Alice,30,text1\r\n"
            "d2,Bob,25,text2\r\n"
        )
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.csv_writer.generate_filename")
    def test_csv_writer_no_header(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "no_header.csv"
        docs = [self._create_doc("d1", "text1", {"name": "Alice"})]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _CsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["doc_id", "name", "text_representation"],
            csv_writer_options={'delimiter': ','},
            write_header=False,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = "d1,Alice,text1\r\n"
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.csv_writer.generate_filename")
    def test_csv_writer_different_delimiter_and_quoting(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "delimiter_quote.csv"
        docs = [self._create_doc("d1", "text1", {"name": "Alice;Extra", "city": "New York"})]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _CsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["doc_id", "name", "city"],
            csv_writer_options={'delimiter': ';', 'quotechar': '"', 'quoting': csv.QUOTE_ALL},
            write_header=True,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = (
            '"doc_id";"name";"city"\r\n'
            '"d1";"Alice;Extra";"New York"\r\n'
        )
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.csv_writer.generate_filename")
    def test_csv_writer_column_selection_and_order(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "column_order.csv"
        docs = [self._create_doc("d1", "text1", {"name": "Alice", "age": 30})]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _CsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["age", "name", "doc_id"],
            csv_writer_options={'delimiter': ','},
            write_header=True,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = (
            "age,name,doc_id\r\n"
            "30,Alice,d1\r\n"
        )
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.csv_writer.generate_filename")
    def test_csv_writer_missing_properties(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "missing_props.csv"
        docs = [
            self._create_doc("d1", "text1", {"name": "Alice", "age": 30}),
            self._create_doc("d2", "text2", {"name": "Bob"}), # Missing "age"
        ]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _CsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["name", "age"],
            csv_writer_options={'delimiter': ','},
            write_header=True,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = (
            "name,age\r\n"
            "Alice,30\r\n"
            "Bob,\r\n"  # Empty string for missing age
        )
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.csv_writer.generate_filename")
    def test_csv_writer_empty_input_records_with_header(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "empty_header.csv"
        serialized_docs = []

        sink = _CsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["name", "age"],
            csv_writer_options={'delimiter': ','},
            write_header=True,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = "name,age\r\n"
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.csv_writer.generate_filename")
    def test_csv_writer_empty_input_records_no_header(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "empty_no_header.csv"
        serialized_docs = []

        sink = _CsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["name", "age"],
            csv_writer_options={'delimiter': ','},
            write_header=False,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = "" # Empty file
        self.assertEqual(content, expected_content)

    def test_csv_writer_init_requires_columns(self):
        mock_plan = MagicMock(spec=Node)
        with self.assertRaisesRegex(ValueError, "columns list must be provided and non-empty for CsvWriter."):
            CsvWriter(plan=mock_plan, path=self.output_path, columns=[])
        
        with self.assertRaisesRegex(ValueError, "columns list must be provided and non-empty for CsvWriter."):
            CsvWriter(plan=mock_plan, path=self.output_path, columns=None) # type: ignore

    def test_csv_writer_get_paths(self):
        mock_plan = MagicMock(spec=Node)
        writer = CsvWriter(
            plan=mock_plan, 
            path=self.output_path, 
            columns=["col1"], 
            filesystem=self.fs
        )

        # Manually create some files in the InMemoryFileSystem
        self.fs.create_dir(self.output_path + "subdir/") # Should be ignored by get_paths (not recursive)
        with self.fs.open_output_stream(self.output_path + "file1.csv") as f:
            f.write(b"test")
        with self.fs.open_output_stream(self.output_path + "file2.txt") as f: # Should be ignored
            f.write(b"test")
        with self.fs.open_output_stream(self.output_path + "file3.csv") as f:
            f.write(b"test")
        with self.fs.open_output_stream(self.output_path + "subdir/file4.csv") as f: # Ignored (in subdir)
            f.write(b"test")


        paths = writer.get_paths()
        expected_paths = sorted([self.output_path + "file1.csv", self.output_path + "file3.csv"])
        self.assertEqual(sorted(paths), expected_paths)


if __name__ == "__main__":
    unittest.main()
