from typing import Callable, Optional

from pyarrow.fs import FileSystem

from sycamore import Context
from sycamore.plan_nodes import Node
from sycamore.data import Document
from sycamore.writers.file_writer import default_doc_to_bytes, default_filename, FileWriter


class DocSetWriter:
    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan

    def opensearch(
        self, *, os_client_args: dict, index_name: str, index_settings: Optional[dict] = None, **resource_args
    ) -> None:
        from sycamore.writers import OpenSearchWriter

        os = OpenSearchWriter(
            self.plan, index_name, os_client_args=os_client_args, index_settings=index_settings, **resource_args
        )
        os.execute()

    def files(
        self,
        path: str,
        filesystem: Optional[FileSystem] = None,
        filename_fn: Callable[[Document], str] = default_filename,
        doc_to_bytes_fn: Callable[[Document], bytes] = default_doc_to_bytes,
        **resource_args
    ) -> None:
        """Writes the content of each Document to a separate file.

        Args:
            path: The path prefix to write to. Should include the scheme if not local.
            filesystem: The pyarrow.fs FileSystem to use.
            filename_fn: A function for generating a file name. Takes a Document
                and returns a unique name that will be appended to path.
            doc_to_bytes_fn: A function from a Document to bytes for generating the data to write.
                Defaults to using text_representation if available, or binary_representation
                if not.
            resource_args: Arguments to pass to the underlying execution environment.
        """
        file_writer = FileWriter(
            self.plan,
            path,
            filesystem=filesystem,
            filename_fn=filename_fn,
            doc_to_bytes_fn=doc_to_bytes_fn,
            **resource_args
        )

        file_writer.execute()
