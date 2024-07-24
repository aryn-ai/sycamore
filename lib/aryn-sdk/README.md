[![PyPI](https://img.shields.io/pypi/v/aryn-sdk)](https://pypi.org/project/aryn-sdk/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aryn-sdk)](https://pypi.org/project/aryn-sdk/)
[![Slack](https://img.shields.io/badge/slack-sycamore-brightgreen.svg?logo=slack)](https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg)
[![Docs](https://readthedocs.org/projects/sycamore/badge/?version=stable)](https://sycamore.readthedocs.io/en/stable/?badge=stable)
![License](https://img.shields.io/github/license/aryn-ai/sycamore)

`aryn-sdk` is a simple client library for interacting with Aryn cloud services.

## Aryn Partitioning Service

Partition pdf files with the Aryn Partitioning Service (APS) through `aryn-sdk`:

```python
from aryn_sdk.partition import partition_file

with open("partition-me.pdf", "rb") as f:
    data = partition_file(
        f,
        use_ocr=True,
        extract_table_structure=True,
        extract_images=True
    )
elements = data['elements']
```

Convert partitioned tables to pandas dataframes for easier use:

```python
from aryn_sdk.partition import partition_file, tables_to_pandas

with open("partition-me.pdf", "rb") as f:
    data = partition_file(
        f,
        use_ocr=True,
        extract_table_structure=True,
        extract_images=True
    )
elements_and_tables = tables_to_pandas(data)
dataframes = [table for (element, table) in elements_and_tables if table is not None]
```
