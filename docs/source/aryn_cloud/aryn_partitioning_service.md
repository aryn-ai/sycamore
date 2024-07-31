# Aryn Partitioning Service


You can use the Aryn Partitioning Service to segment PDF's into labeled bounding boxes identifying titles, tables, table rows and columns, images, and regular text. Bounding boxes are returned as JSON with their associated text for easy use.

There are three ways to use the Aryn Partitioning Service: through the `ArynPartitioner`, through the `aryn-sdk` client, and directly from the HTTP service.

To follow along below, we will need an Aryn API key, which we can get at [aryn.ai/get-started](https://www.aryn.ai/get-started). You will recieve the API key in your email inbox.

## Using with Sycamore's Partition transform

Say you have a set of pdfs located at the path stored in `work_dir`. We partition these documents with the code snippet below:

```python
aryn_api_key = "PUT API KEY HERE"

ctx = sycamore.init()
pdf_docset = context.read.binary(work_dir, binary_format="pdf")
partitioned_docset = pdf_docset.partition(ArynPartitioner(aryn_api_key=aryn_api_key))
```
Alternatively, we can store our Aryn API key at `~/.aryn/config.yaml` like so:
```yaml
aryn_token: "PUT API KEY HERE"
```
Which makes our Sycamore script the following:
```python
ctx = sycamore.init()
pdf_docset = context.read.binary(work_dir, binary_format="pdf")
partitioned_docset = pdf_docset.partition(ArynPartitioner())
```

## Using `aryn-sdk`

The `aryn-sdk` client is a thin python library that calls the Aryn Partitioning Service and provides a few utility methods around it. Install the `aryn-sdk` client with `pip install aryn-sdk`.
Partition a document like so:

```python
from aryn_sdk.partition import partition_file
with open("mydocument.pdf", "rb") as f:
    data = partition_file(f)
```

`partition_file` takes the same options as curl, except as keyword arguments. You can find a list of these options [here](###Specifying Options).

### Key management

By default, `aryn-sdk` looks for Aryn API keys first in the environment variable `ARYN_API_KEY`, and then in `~/.aryn/config.yaml`. You can override this behavior by specifying a key directly or a different path to the Aryn config file:
```python
from aryn_sdk.partition import partition_file
from aryn_sdk.config import ArynConfig
with open("mydocument.pdf", "rb") as f:
    data = partition_file(f, aryn_api_key="YOUR-API-KEY")
with open("mydocument.pdf", "rb") as f:
    data = partition_file(f, aryn_config=ArynConfig(aryn_config_path="~/dotfiles/.aryn/config.yaml"))
```

### Helper Functions

`aryn_sdk` provides some helper functions to make working with and visualizing the output of `partition_file` easier.

```python
from aryn_sdk.partition import partition_file, table_elem_to_dataframe, draw_with_boxes
with open("mydocument.pdf", "rb") as f:
    data = partition_file(f, extract_table_structure=True, use_ocr=True, extract_images=True, threshold=0.35)

# Produce a pandas DataFrame representing one of the extracted tables
table_elements = [elt for elt in data['elements'] if elt['type'] == 'table']
dataframe = table_elem_to_dataframe(table_elements[0])

# Draw the detected bounding boxes on the pages. requires poppler
images = draw_with_boxes("mydocument.pdf", data)
```


## Using `curl`

`curl` an example document to use with the partitioning service if you do not have one already.
```bash
curl http://arxiv.org/pdf/1706.03762.pdf -o document.pdf
```
Change `PUT API KEY HERE` below to your Aryn API key. If you have a different document, change `@document.pdf` to `@/path/to/your/document.pdf` below.
```bash
export ARYN_API_KEY="PUT API KEY HERE"
curl "https://api.aryn.cloud/v1/document/partition" -o "document.json" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.pdf";
```
Your results have been saved to `document.json`. `cat` your document to see the results.
```bash
cat document.json
```

### Specifying Options

We can extract the table structure from our document with the following command. Make sure to use double quotes in `options`.

```bash
export ARYN_API_KEY="PUT API KEY HERE"
curl "https://api.aryn.cloud/v1/document/partition" -o "document.json" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.pdf" -F "options={\"extract_table_structure\": true}";
```

The available options are listed below:

* ```threshold```: The threshold to use for accepting the models predicted bounding boxes. A lower value will include more objects, but may have overlaps, a higher value will reduce the number of overlaps, but may miss legitimate objects. It defaults to ```0.4```.
* ```use_ocr```: If ```true```, the partitioner uses OCR to extract text from the PDF. It defaults to ```false```, where the partitioner attempts to directly extract the text from the underlying PDF in the bounding box. It currently uses Tesseract for extraction.
* `extract_table_structure`: If `true`, the partitioner runs a separate table extraction model to extract cells from regions of the document identified as tables.
* `extract_images`: If `true`, the partitioner crops each region identified as an image and attaches it to the associated `ImageElement`. This can later be fed into the `SummarizeImages` transform when used within Sycamore.

You can use multiple options at the same time like in the example below:

```bash
export ARYN_API_KEY="PUT API KEY HERE"
curl "https://api.aryn.cloud/v1/document/partition" -o "document.json" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.pdf" -F "options={\"extract_table_structure\": true, \"threshold\": 0.2}";
```
