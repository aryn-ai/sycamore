import unittest
import csv
from unittest.mock import MagicMock, patch
import pyarrow.fs as pa_fs

# If _CsvBlockDataSink is intended to be private, this import might need adjustment
# For now, assume it's importable for direct testing as per instructions.
from sycamore.connectors.file.tsv_writer import TsvWriter, _TsvBlockDataSink
from sycamore.connectors.file.file_writer_ray import generate_filename # Import for potential mocking
from sycamore.data import Document
from sycamore.plan_nodes import Node


class TestTsvWriter(unittest.TestCase):
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

    @patch("sycamore.connectors.file.tsv_writer.generate_filename")
    def test_tsv_writer_basic_write(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "part-000000.tsv"
        
        docs = [
            self._create_doc("d1", "text1", {"name": "Alice", "age": 30}),
            self._create_doc("d2", "text2", {"name": "Bob", "age": 25}),
        ]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _TsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["doc_id", "name", "age", "text_representation"],
            tsv_writer_options={'delimiter': '\t'}, # Default for TsvWriter, explicit here for clarity
            write_header=True,
        )
        
        written_path = sink._write_single_block(serialized_docs, 0)
        self.assertEqual(written_path, self.output_path + "part-000000.tsv")

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = (
            "doc_id\tname\tage\ttext_representation\r\n"
            "d1\tAlice\t30\ttext1\r\n"
            "d2\tBob\t25\ttext2\r\n"
        )
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.tsv_writer.generate_filename")
    def test_tsv_writer_no_header(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "no_header.tsv"
        docs = [self._create_doc("d1", "text1", {"name": "Alice"})]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _TsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["doc_id", "name", "text_representation"],
            tsv_writer_options={'delimiter': '\t'},
            write_header=False,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = "d1\tAlice\ttext1\r\n"
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.tsv_writer.generate_filename")
    def test_tsv_writer_quoting_if_needed(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "quoting.tsv"
        # Property "name" contains a tab, "city" contains a newline.
        docs = [self._create_doc("d1", "text1", {"name": "Alice\tExtra", "city": "New\nYork"})]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _TsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["doc_id", "name", "city"],
            # csv.QUOTE_MINIMAL is the default, which should quote fields containing delimiter or quotechar
            tsv_writer_options={'delimiter': '\t', 'quoting': csv.QUOTE_MINIMAL},
            write_header=True,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        # Expect fields with tabs or newlines to be quoted.
        # Default quotechar is '"'.
        expected_content = (
            "doc_id\tname\tcity\r\n"
            'd1\t"Alice\tExtra"\t"New\nYork"\r\n'
        )
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.tsv_writer.generate_filename")
    def test_tsv_writer_column_selection_and_order(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "column_order.tsv"
        docs = [self._create_doc("d1", "text1", {"name": "Alice", "age": 30})]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _TsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["age", "name", "doc_id"],
            tsv_writer_options={'delimiter': '\t'},
            write_header=True,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = (
            "age\tname\tdoc_id\r\n"
            "30\tAlice\td1\r\n"
        )
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.tsv_writer.generate_filename")
    def test_tsv_writer_missing_properties(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "missing_props.tsv"
        docs = [
            self._create_doc("d1", "text1", {"name": "Alice", "age": 30}),
            self._create_doc("d2", "text2", {"name": "Bob"}), # Missing "age"
        ]
        serialized_docs = [doc.serialize() for doc in docs]

        sink = _TsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["name", "age"],
            tsv_writer_options={'delimiter': '\t'},
            write_header=True,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = (
            "name\tage\r\n"
            "Alice\t30\r\n"
            "Bob\t\r\n" 
        )
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.tsv_writer.generate_filename")
    def test_tsv_writer_empty_input_records_with_header(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "empty_header.tsv"
        serialized_docs = []

        sink = _TsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["name", "age"],
            tsv_writer_options={'delimiter': '\t'},
            write_header=True,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = "name\tage\r\n"
        self.assertEqual(content, expected_content)

    @patch("sycamore.connectors.file.tsv_writer.generate_filename")
    def test_tsv_writer_empty_input_records_no_header(self, mock_generate_filename):
        mock_generate_filename.return_value = self.output_path + "empty_no_header.tsv"
        serialized_docs = []

        sink = _TsvBlockDataSink(
            path=self.output_path,
            filesystem=self.fs,
            columns=["name", "age"],
            tsv_writer_options={'delimiter': '\t'},
            write_header=False,
        )
        written_path = sink._write_single_block(serialized_docs, 0)

        with self.fs.open_input_stream(written_path) as f:
            content = f.read().decode("utf-8")
        
        expected_content = "" 
        self.assertEqual(content, expected_content)

    def test_tsv_writer_init_requires_columns(self):
        mock_plan = MagicMock(spec=Node)
        with self.assertRaisesRegex(ValueError, "columns list must be provided and non-empty for TsvWriter."):
            TsvWriter(plan=mock_plan, path=self.output_path, columns=[])
        
        with self.assertRaisesRegex(ValueError, "columns list must be provided and non-empty for TsvWriter."):
            TsvWriter(plan=mock_plan, path=self.output_path, columns=None) # type: ignore

    def test_tsv_writer_get_paths(self):
        mock_plan = MagicMock(spec=Node)
        writer = TsvWriter(
            plan=mock_plan, 
            path=self.output_path, 
            columns=["col1"], 
            filesystem=self.fs
        )

        self.fs.create_dir(self.output_path + "subdir/") 
        with self.fs.open_output_stream(self.output_path + "file1.tsv") as f:
            f.write(b"test")
        with self.fs.open_output_stream(self.output_path + "file2.txt") as f: 
            f.write(b"test")
        with self.fs.open_output_stream(self.output_path + "file3.tsv") as f:
            f.write(b"test")
        with self.fs.open_output_stream(self.output_path + "subdir/file4.tsv") as f: 
            f.write(b"test")

        paths = writer.get_paths()
        expected_paths = sorted([self.output_path + "file1.tsv", self.output_path + "file3.tsv"])
        self.assertEqual(sorted(paths), expected_paths)


if __name__ == "__main__":
    unittest.main()
