[![PyPI](https://img.shields.io/pypi/v/aryn-sdk)](https://pypi.org/project/aryn-sdk/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aryn-sdk)](https://pypi.org/project/aryn-sdk/)
[![Slack](https://img.shields.io/badge/slack-sycamore-brightgreen.svg?logo=slack)](https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg)
[![Docs](https://readthedocs.org/projects/sycamore/badge/?version=stable)](https://sycamore.readthedocs.io/en/stable/?badge=stable)
![License](https://img.shields.io/github/license/aryn-ai/sycamore)

`aryn-sdk` is a simple client library for interacting with Aryn cloud services.

## Aryn DocParse

Partition pdf files with Aryn DocParse through `aryn-sdk`:

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

Convert a partitioned table element to a pandas dataframe for easier use:

```python
from aryn_sdk.partition import partition_file, table_elem_to_dataframe

with open("partition-me.pdf", "rb") as f:
    data = partition_file(
        f,
        use_ocr=True,
        extract_table_structure=True,
        extract_images=True
    )

# Find the first table and convert it to a dataframe
df = None
for element in data['elements']:
    if element['type'] == 'table':
        df = table_elem_to_dataframe(element)
        break
```

Or convert all partitioned tables to pandas dataframes in one shot:

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

Visualize partitioned documents by drawing on the bounding boxes:

```python
from aryn_sdk.partition import partition_file, draw_with_boxes

with open("partition-me.pdf", "rb") as f:
    data = partition_file(
        f,
        use_ocr=True,
        extract_table_structure=True,
        extract_images=True
    )
page_pics = draw_with_boxes("partition-me.pdf", data, draw_table_cells=True)

from IPython.display import display
display(page_pics[0])
```

> Note: visualizing documents requires `poppler`, a pdf processing library, to be installed. Instructions for installing poppler can be found [here](https://pypi.org/project/pdf2image/)

Convert image elements to more useful types, like PIL, or image format typed byte strings

```python
from aryn_sdk.partition import partition_file, convert_image_element

with open("my-favorite-pdf.pdf", "rb") as f:
    data = partition_file(
        f,
        extract_images=True
    )
image_elts = [e for e in data['elements'] if e['type'] == 'Image']

pil_img = convert_image_element(image_elts[0])
jpg_bytes = convert_image_element(image_elts[1], format='JPEG')
png_str = convert_image_element(image_elts[2], format="PNG", b64encode=True)
```

### Async Aryn DocParse

#### Single Job Example
```python
import time
from aryn_sdk.partition import partition_file_async_submit, partition_file_async_result

with open("my-favorite-pdf.pdf", "rb") as f:
    response = partition_file_async_submit(
        f,
        use_ocr=True,
        extract_table_structure=True,
    )

job_id = response["job_id"]

# Poll for the results
while True:
    result = partition_file_async_result(job_id)
    if result["status"] != "pending":
        break
    time.sleep(5)
```

Optionally, you can also set a webhook for Aryn to call when your job is completed:

```python
partition_file_async_submit("path/to/my/file.docx", webhook_url="https://example.com/alert")
```

Aryn will POST a request containing a body like the below:
```json
{"done": [{"job_id": "aryn:j-47gpd3604e5tz79z1jro5fc"}]}
```

#### Multi-Job Example

```python
import logging
import time
from aryn_sdk.partition import partition_file_async_submit, partition_file_async_result

files = [open("file1.pdf", "rb"), open("file2.docx", "rb")]
job_ids = [None] * len(files)
for i, f in enumerate(files):
    try:
        job_ids[i] = partition_file_async_submit(f)["job_id"]
    except Exception as e:
        logging.warning(f"Failed to submit {f}: {e}")

results = [None] * len(files)
for i, job_id in enumerate(job_ids):
    while True:
        result = partition_file_async_result(job_id)
        if result["status"] != "pending":
            break
        time.sleep(5)
    results[i] = result
```

#### Cancelling an async job

```python
from aryn_sdk.partition import partition_file_async_submit, partition_file_async_cancel
        job_id = partition_file_async_submit(
                    "path/to/file.pdf",
                    use_ocr=True,
                    extract_table_structure=True,
                    extract_images=True,
                )["job_id"]

        partition_file_async_cancel(job_id)
```

#### List pending jobs

```
from aryn_sdk.partition import partition_file_async_list
partition_file_async_list()
```
