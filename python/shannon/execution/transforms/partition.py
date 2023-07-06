from ray.data import Dataset
from shannon.execution import (Node, Transform)
from shannon.execution.kernels import UnstructuredPartitionPdfKernel
from typing import Optional


class Partition(Transform):
    def __init__(
            self,
            child: Node,
            col_name: str,
            max_partition: Optional[int],
            **resource_args):
        super().__init__(child, **resource_args)
        self.col_name = col_name
        self.max_partition = max_partition

    def set_max_partition(self, max_partition) -> None:
        self.max_partition = max_partition


class PartitionPDF(Partition):
    def __init__(
            self,
            child: Node,
            col_name: str,
            include_page_breaks: bool = False,
            strategy: str = "auto",
            infer_table_structure: bool = False,
            ocr_languages: str = "eng",
            max_partition: Optional[int] = None,
            include_metadata: bool = True,
            **resource_args):
        super().__init__(
            child, max_partition=max_partition,
            col_name=col_name, **resource_args)
        self.include_page_breaks = include_page_breaks
        self.strategy = strategy
        self.infer_table_structure = infer_table_structure
        self.ocr_languages = ocr_languages
        self.include_metadata = include_metadata

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        partitioner = UnstructuredPartitionPdfKernel(
            col_name=self.col_name,
            include_page_breaks=self.include_page_breaks,
            strategy=self.strategy,
            infer_table_structure=self.infer_table_structure,
            ocr_languages=self.ocr_languages,
            max_partition=self.max_partition,
            include_metadata=self.include_metadata)
        dataset = input_dataset.flat_map(partitioner.partition)
        return dataset
