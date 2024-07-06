# Aryn Partitioning Service


You can use the Aryn Partitioning Service to segment PDF's into labeled bounding boxes identifying titles, tables, table rows and columns, images, and regular text.

There are two ways to use the Aryn Partitioning Service: through the `ArynPartitioner` and directly from the HTTP service.

To follow along below, we will need an Aryn API key, which we can get at [aryn.ai/cloud](https://www.aryn.ai/cloud). You will recieve the API key in your email inbox.
## Using Aryn Partitioner

Say you have a set of pdfs located at the path stored in `work_dir`, and a manifest of them at `manifest_path`. We partition these documents with the code snippet below:

```python
aryn_api_key = "PUT API KEY HERE"

ctx = sycamore.init()
pdf_docset = context.read.binary(work_dir, binary_format="pdf", metadata_provider=JsonManifestMetadataProvider(manifest_path))
partitioned_docset = pdf_docset.partition(ArynPartitioner(aryn_api_key=aryn_api_key))
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
curl "https://api.aryn.cloud/v1/document/partition" -o "document.json" -H "Authorization: Bearer $ARYN_TOKEN" -F "pdf=@document.pdf" -F "options={\"extract_table_structure\": true}";
```

The available options are listed below:

* ```threshold```: The threshold to use for accepting the models predicted bounding boxes. A lower value will include more objects, but may have overlaps, a higher value will reduce the number of overlaps, but may miss legitimate objects. It defaults to ```0.4```.
* ```use_ocr```: If ```true```, the partitioner uses OCR to extract text from the PDF. It defaults to ```false```, where the partitioner attempts to directly extract the text from the underlying PDF in the bounding box. It currently uses Tesseract for extraction.
* `extract_table_structure`: If `true`, the partitioner runs a separate table extraction model to extract cells from regions of the document identified as tables.
* `extract_images`: If `true`, the partitioner crops each region identified as an image and attaches it to the associated `ImageElement`. This can later be fed into the `SummarizeImages` transform.