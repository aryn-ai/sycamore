from ray.data import (Dataset, read_binary_files, read_json)
from ray.data.datasource import FileExtensionFilter
from shannon.execution import Scan
from typing import (List, Optional, Union)


class FileScan(Scan):
    """A base scan class for file based data"""
    def __init__(
            self,
            paths: Union[str, List[str]],
            *,
            parallelism: Optional[int] = None,
            **resource_args):
        super().__init__(**resource_args)
        self._paths = paths
        self.parallelism = parallelism


class BinaryScan(FileScan):
    """Scan data file into raw bytes

    For each file, BinaryScan creates one doc in the form of {"bytes", binary},
    this would serve as the downstream of following workload.
    """
    def __init__(
            self,
            paths: Union[str, List[str]],
            *,
            binary_format: str,
            parallelism: Optional[int] = None,
            **resource_args):
        super().__init__(paths, parallelism=parallelism, **resource_args)
        self._paths = paths
        self.parallelism = -1 if parallelism is None else parallelism
        self._binary_format = binary_format

    def execute(self) -> "Dataset":
        partition_filter = FileExtensionFilter(self.format())
        return read_binary_files(
            self._paths,
            parallelism=self.parallelism,
            partition_filter=partition_filter,
            ray_remote_args=self.resource_args)

    def format(self):
        return self._binary_format


class JsonScan(FileScan):
    def __init__(
            self,
            paths: Union[str, List[str]],
            *,
            parallelism: Optional[int] = None,
            **resource_args):
        super().__init__(
            paths, parallelism=parallelism, **resource_args)

    def execute(self) -> "Dataset":
        return read_json(
            paths=self._paths,
            parallelism=self.parallelism,
            **self.resource_args)

    def format(self):
        return "json"
