# Aryn Partitioning Service


You can use the Aryn Partitioning Service to segment PDF's into labeled bounding boxes identifying titles, tables, images, and regular text.

There are two ways to use the Aryn Partitioning Service: through the `ArynPartitioner` and directly through HTTP. `ArynPartitioner` is like a `SycamorePartitioner` that runs some computation remotely.

We will need an aryn token, which we can get at [aryn.ai/cloud](https://www.aryn.ai/cloud). You will recieve a token in your email inbox.

## Using ArynPartitioner

Say you have a set of pdfs located at the path stored in `work_dir`, and a manifest of them at `manifest_path`. We partition these documents in the code snippet below:

```python
aryn_token = "PUT TOKEN HERE"

ctx = sycamore.init()
pdf_docset = context.read.binary(work_dir, binary_format="pdf", metadata_provider=JsonManifestMetadataProvider(manifest_path))
partitioned_docset = pdf_docset.partition(ArynPartitioner(aryn_token=aryn_token))
```

## Using `curl`

`curl` an example document to use with the partitioning service if you do not have one already.
```bash
curl http://arxiv.org/pdf/1706.03762.pdf -o document.pdf
```
Change `PUT TOKEN HERE` below to your token. If you have a different document, change `@document.pdf` to `@/path/to/your/document.pdf` below.
```bash
export ARYN_TOKEN="PUT TOKEN HERE"
curl "https://api.aryn.cloud/v1/document/partition" -o "document.json" -H "Authorization: Bearer $ARYN_TOKEN" -F "pdf=@document.pdf";
```
Your results have been saved to `document.json`. `cat` your document to see the results.
```bash
cat document.json
```

### Specifying Options

We can extract the table structure from our document with the following command. Make sure to use double quotes in `options`.

```bash
export ARYN_TOKEN="PUT TOKEN HERE"
curl "https://api.aryn.cloud/v1/document/partition" -o "document.json" -H "Authorization: Bearer $ARYN_TOKEN" -F "pdf=@document.pdf" -F "options={\"extract_table_structure\": true}";
```

The available options are listed below:

* ```model_name_or_path```: The HuggingFace coordinates or model local path. It defaults to ```SYCAMORE_DETR_MODEL```, and you should only change it if you are testing a custom model. * * ```threshold```: The threshold to use for accepting the models predicted bounding boxes. A lower value will include more objects, but may have overlaps, a higher value will reduce the number of overlaps, but may miss legitimate objects. It defaults to ```0.4```.
* ```use_ocr```: If ```true```, the partitioner uses OCR to extract text from the PDF. It defaults to ```false```, where the partitioner attempts to directly extract the text from the underlying PDF in the bounding box. It currently uses Tesseract for extraction.
* ```ocr_images```: If set to ```true``` alongside ```use_ocr```, the partitioner will attempt to OCR regions of the document identified as images.
* ```ocr_tables```: If set to ```true``` alongside ```use_ocr```, the partitioner will attempt to OCR regions on the document identified as tables. This should not be set when `extract_table_structure` is ```true```. It currently uses EasyOCR for extraction.
* `extract_table_structure`: If `true`, the partitioner runs a separate table extraction model to extract cells from regions of the document identified as tables. Do not set if 'ocr_tables' is true.
* `extract_images`: If `true`, the partitioner crops each region identified as an image and attaches it to the associated `ImageElement`. This can later be fed into the `SummarizeImages` transform.