import csv
import io
from typing import Any, List, Optional, Dict

import ray
from ray.data.block import Block
from ray.data.datasource import DataSink
from ray.types import ObjectRef
from pyarrow.fs import FileSystem
import pyarrow as pa

from sycamore.data import Document # Assuming Document has a .serialize() method that returns a dict
from sycamore.plan_nodes import Node, Write
from sycamore.connectors.file.file_writer_ray import generate_filename


class _TsvBlockDataSink(DataSink):
    def __init__(
        self,
        path: str,
        filesystem: Optional[FileSystem],
        columns: List[str],
        tsv_writer_options: dict,
        write_header: bool,
    ):
        self.path = path
        self.filesystem = filesystem or pa.fs.LocalFileSystem()
        self.columns = columns
        self.tsv_writer_options = tsv_writer_options
        self.write_header = write_header
        # Ensure filesystem is initialized for generate_filename if it uses it internally
        if self.filesystem and hasattr(self.filesystem, "create_dir"):
             self.filesystem.create_dir(self.path, recursive=True)


    def write(self, blocks: List[ObjectRef[Block]], ctx) -> List[ObjectRef[Any]]:
        results = []
        for i, block_ref in enumerate(blocks):
            # Assuming block_ref is an ObjectRef to a pyarrow.Table or list of records
            # If blocks are not already pyarrow Tables, conversion might be needed.
            # For now, assuming ray.get(block_ref) gives something processable.
            # The structure of data in `block` depends on what the upstream Dataset.map produces.
            # Let's assume it's a list of dicts (serialized Documents) as per typical Sycamore flow.
            # Or if it's Arrow, we need to handle pa.Table.
            
            # If blocks are Arrow Tables, and 'doc' column contains serialized Document
            # block_data = ray.get(block_ref) # This could be pa.Table
            # if not isinstance(block_data, pa.Table):
            #    raise ValueError("Expected block to be a PyArrow Table.")
            # num_rows = block_data.num_rows
            # records = [block_data.column("doc")[j].as_py() for j in range(num_rows)]

            # If blocks are lists of already serialized Python objects (dicts)
            records = ray.get(block_ref) # Expect List[Dict]
            if not isinstance(records, list): # Basic check
                 # If it's a PyArrow block, convert to list of dicts
                if isinstance(records, pa.Table):
                    # Assuming the table has one column named 'doc' which holds the serialized document
                    # This aligns with how JsonWriter prepares data if map(lambda d: {"doc": d.serialize()}) is used.
                    # If the table directly contains the document fields, logic needs to change.
                    if "doc" in records.column_names:
                         processed_records = []
                         for i_row in range(records.num_rows):
                            doc_data = records.column("doc")[i_row].as_py()
                            # Ensure doc_data is a dict (serialized Document)
                            if not isinstance(doc_data, dict):
                                raise ValueError(f"Expected 'doc' column to contain dict, got {type(doc_data)}")
                            processed_records.append(doc_data)
                         records = processed_records
                    else: # Table has columns directly, try to convert row by row to dict
                        records = records.to_pylist()

                else:
                    raise ValueError(f"Expected block to be a list of dicts or PyArrow Table, got {type(records)}")

            if not records: # Skip empty blocks
                results.append("empty_block") # Or some other placeholder
                continue

            # Use task_idx from ctx for unique filenames if available, otherwise fallback
            writer_block_id = ctx.task_idx if ctx and hasattr(ctx, "task_idx") else i
            results.append(self._write_single_block(records, writer_block_id))
        return results

    def _write_single_block(self, records: List[Dict[str, Any]], writer_block_id: int) -> str:
        block_path = generate_filename(self.path, writer_block_id, "tsv")

        string_buffer = io.StringIO()
        # Ensure dialect-specific options like quotechar, quoting are passed correctly
        writer = csv.writer(string_buffer, **self.tsv_writer_options)

        if self.write_header:
            writer.writerow(self.columns)

        for record_dict in records: # Assuming record_dict is a serialized Document
            row_values = []
            # Serialized Document structure:
            # {
            #   "doc_id": "...", "type": "...", "text_representation": "...",
            #   "elements": [...], "embedding": [...],
            #   "properties": {"prop1": "val1", "path": "/data/file.txt"}
            # }
            current_doc_properties = record_dict.get("properties", {})
            text_representation = record_dict.get("text_representation")

            for col_name in self.columns:
                if col_name == "text_representation": # Special handling for text_representation
                    row_values.append(text_representation if text_representation is not None else "")
                elif col_name == "doc_id":
                    row_values.append(record_dict.get("doc_id", ""))
                elif col_name == "type":
                    row_values.append(record_dict.get("type", ""))
                # Add other top-level fields if they can be columns: elements, embedding
                elif col_name in current_doc_properties:
                    row_values.append(current_doc_properties[col_name])
                else:
                    # Fallback for fields not in properties or special cased
                    row_values.append(record_dict.get(col_name, ""))
            writer.writerow(row_values)

        csv_data = string_buffer.getvalue().encode("utf-8")

        if not self.filesystem:
            raise RuntimeError("Filesystem not initialized in _TsvBlockDataSink")

        with self.filesystem.open_output_stream(block_path) as f:
            f.write(csv_data)
        return block_path


class TsvWriter(Write):
    def __init__(
        self,
        plan: Node,
        path: str,
        *,
        filesystem: Optional[FileSystem] = None,
        columns: List[str],
        tsv_writer_options: Optional[dict] = None,
        write_header: bool = True,
        **ray_remote_args,
    ):
        super().__init__(plan, **ray_remote_args)
        self.path = path
        self._filesystem = filesystem or pa.fs.LocalFileSystem()
        if not columns:
            raise ValueError("columns list must be provided and non-empty for TsvWriter.")
        self.columns = columns
        self._tsv_writer_options = {"delimiter": "\t"}
        if tsv_writer_options:
            self._tsv_writer_options.update(tsv_writer_options)
        self.write_header = write_header

    def execute(self, **kwargs) -> None: # write_datasink typically returns None or a list of WriteTaskStatus
        # Map to serialized documents first, as the sink expects dicts
        # This aligns with JsonWriter's pattern: input_dataset.map(lambda d: {"doc": d.serialize()})
        # However, our _CsvBlockDataSink now handles various input types (List[Dict] or pa.Table)
        # If the input dataset is already Dataset[Document], we need to serialize it.
        # If it's Dataset[Dict], it might already be serialized.
        
        input_dataset = self.child().execute(**kwargs) # Should be Dataset[Document]
        
        # If the sink expects each record to be a dict representing a Document,
        # we should .map(lambda doc: doc.serialize())
        # The current _CsvBlockDataSink implementation tries to handle pa.Table with a 'doc' column
        # or a list of dicts (serialized docs).
        # Let's ensure the data is in List[Dict[str, Any]] format per block for simplicity.
        # If input_dataset contains Document objects, they need to be serialized.
        # A common pattern is to have the sink process dicts that are Document.serialize() output.
        
        # Option 1: Dataset contains Document objects. Serialize them.
        # serialized_dataset = input_dataset.map(lambda doc_obj: doc_obj.serialize())

        # Option 2: Dataset might already be dicts, or Arrow table.
        # The current sink tries to handle this. For robustness, let's assume
        # we want to process Document objects and serialize them before writing.
        
        # Let's assume the input_dataset from child().execute() is Dataset[Document]
        # We need to transform it into a Dataset of dicts (serialized documents)
        # so that _CsvBlockDataSink receives blocks of these dicts.
        def serialize_doc(doc: Document) -> Dict[str, Any]:
            return doc.serialize()

        # This map ensures that each item in the dataset (and thus in each block)
        # is a dictionary, which is what _TsvBlockDataSink._write_single_block expects.
        dict_dataset = input_dataset.map(serialize_doc)

        datasink = _TsvBlockDataSink(
            path=self.path,
            filesystem=self._filesystem,
            columns=self.columns,
            tsv_writer_options=self._tsv_writer_options,
            write_header=self.write_header,
        )
        
        # write_datasink will handle the execution of the sink over the dataset.
        # It typically returns None or a list of statuses, not the dataset itself.
        dict_dataset.write_datasink(datasink, ray_remote_args=self.ray_remote_args)

    def get_paths(self) -> List[str]:
        if self._filesystem is None:
            return []
        try:
            file_infos = self._filesystem.get_file_info(pa.fs.FileSelector(self.path, recursive=False))
            return [fi.path for fi in file_infos if fi.type == pa.fs.FileType.File and fi.base_name.endswith(".tsv")]
        except FileNotFoundError:
            return []
