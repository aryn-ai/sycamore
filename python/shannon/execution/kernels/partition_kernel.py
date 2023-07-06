import io
from typing import (Any, Dict, List, Optional)
from unstructured.partition.pdf import partition_pdf


class PartitionKernel:
    def partition(self, binary: Dict[str, bytes]) -> List[Dict[str, Any]]:
        pass


class UnstructuredPartitionPdfKernel(PartitionKernel):

    def __init__(
            self,
            col_name: str,
            include_page_breaks: bool = False,
            strategy: str = "auto",
            infer_table_structure: bool = False,
            ocr_languages: str = "eng",
            max_partition: Optional[int] = None,
            include_metadata: bool = True):
        self._col_name = col_name
        self._include_page_breaks = include_page_breaks
        self._strategy = strategy
        self._infer_table_structure = infer_table_structure
        self._ocr_languages = ocr_languages
        self._max_partition = max_partition
        self._include_metadata = include_metadata

    def partition(self, doc: Dict[str, bytes]) -> List[Dict[str, Any]]:
        bytes_io = io.BytesIO(doc[self._col_name])
        text = [{self._col_name: str(element)} for element in partition_pdf(
            file=bytes_io, include_page_breaks=self._include_page_breaks,
            strategy=self._strategy,
            infer_table_structure=self._infer_table_structure,
            ocr_languages=self._ocr_languages,
            max_partition=self._max_partition,
            include_metadata=self._include_metadata)]
        return text
