(Partition)=
## Partition
To process raw documents and bring them into DocSets, Sycamore must first segment and chunk the document and label each element, such as headings, tables, and figures. This process is called document partitioning, and it is carried out by the Partition transform. Partitioners are format specific, and we recommend:

* PDF: Aryn Partitioner
* HTML: Html Partitioner

Sycamore also includes the ``UnstructuredPdfPartitioner`` for PDFs as well.

```python
 partitioned_docset = docset.partition(partitioner=ArynPartitioner())
```

### Aryn Partitioner

The Aryn Partitioner was built from the ground-up for high-quality segmentation using a new AI vision model at it's core. This model is [a Deformable DEtection Transformer (DETR) model](https://huggingface.co/Aryn/deformable-detr) trained on [DocLayNet](https://github.com/DS4SD/DocLayNet), an open source, human-annotated document layout segmentation dataset. This model is 100% open source with an Apache v2.0 license.

There are several options you can use in the Aryn Partitioner for table extraction, OCR, and more.

Parameters:

* ```use_partitioning_service```: If ```True```, the partitioner uses *Aryn DocParse* (formerly known as the Aryn Partitioning Service). Defaults to ```True```. For see options for the service, see the [Aryn Partitioning Service](https://docs.aryn.ai/docparse/processing_options) docs.
* ```model_name_or_path```: The HuggingFace coordinates or model local path. It defaults to ```SYCAMORE_DETR_MODEL```, and you should only change it if you are testing a custom model. Ignored when ```use_partitioning_service``` is ```True```.
* ``threshold``: This represents the threshold for accepting the model’s predicted bounding boxes. It defaults to “auto”, where the service uses a processing method to find the best prediction for each possible bounding box. This is the recommended setting. However, you can override this by specifying a numerical threshold between 0 and 1. If you specify a numerical threshold, only bounding boxes with confidence scores higher than the threshold will be returned (instead of using the processing method described above). A lower value will include more objects, but may have overlaps, while a higher value will reduce the number of overlaps, but may miss legitimate objects. If you do set the threshold manually, we recommend starting with a value of 0.32.
* ``ocr_model``: model to use for OCR. Choices are "easyocr", "paddle", "tesseract" and "legacy", which correspond to EasyOCR, PaddleOCR, and Tesseract respectively, with "legacy" being a combination of Tesseract for text and EasyOCR for tables. If you choose paddle make sure to install paddlepaddle or paddlepaddle-gpu depending on whether you have a CPU or GPU. Further details are found at: https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html. Note: this will be ignored for Aryn DocParse, which uses its own OCR implementation. Defaults to ``easyocr``.
* ```use_ocr```: If ```True```, the partitioner uses OCR to extract text from the PDF. It defaults to ```False```, where the partitioner attempts to directly extract the text from the underlying PDF in the bounding box. It currently uses Tesseract for extraction.
* ```per_element_ocr```: If true, will run OCR on each element individually instead of the entire page. Note: this
    will be ignored for Aryn DocParse, which uses its own OCR implementation. default: True
* `extract_table_structure`: If `True`, the partitioner runs a separate table extraction model to extract cells from regions of the document identified as tables.
* `table_structure_extractor`: The table extraction implementation to use when `extract_table_structure` is `True`. The default is the `TableTransformerStructureExtractor`. Ignored when ```use_partitioning_service``` is ```True```.
* `extract_images`: If `True`, the partitioner crops each region identified as an image and attaches it to the associated `ImageElement`. This can later be fed into the `SummarizeImages` transform.
* `aryn_api_key`: The account token used to authenticate with Aryn's servers.
* `aryn_partitioner_address`: The address of the server to use to partition the document
* `use_cache`: Cache results from the partitioner for faster inferences on the same documents in future runs. default: False
* `pages_per_call`: Number of pages to send in a single call to the remote service. Default is -1, which means send all pages in one call.
* `output_format`: controls output representation: json (default) or markdown.
* `text_extraction_options`: Dict of options that are sent to the TextExtractor implementation, either pdfminer or OCR. Currently supports the 'object_type' property for pdfminer, which can be set to 'boxes' or 'lines' to control the granularity of output. Note that this has a separate implementation for the Aryn DocParse service.
* `output_label_options`: A dictionary for configuring output label behavior. It supports two options: ``title_candidate_elements``, a list of strings representing the label types allowed to be promoted to a title. ``promote_title``, a boolean specifying whether to pick the largest element by font size on the first page from among the elements on that page that have one of the types specified in title_candidate_elements and promote it to type "Title" if there is no element on the first page of type "Title" already. Here is an example set of output label options: ``{"promote_title": True, "title_candidate_elements": ["Section-header", "Caption"]}``. default: None (no element is promoted to "Title").
* `**kwargs`: Additional keyword arguments to pass to the remote partitioner. See the [Aryn Partitioning Service](https://docs.aryn.ai/docparse/processing_options) docs for more information.


Here is an example of chunking and using table extraction:

```Python
ctx = sycamore.init()
docset = ctx.read.binary(s3://my-bucket/my-folder/, binary_format="pdf")
            .partition(partitioner=ArynPartitioner(extract_table_structure=True))
```

Here is an example of chunking and using OCR:

```Python
ctx = sycamore.init()
docset = ctx.read.binary(s3://my-bucket/my-folder/, binary_format="pdf")
            .partition(partitioner=ArynPartitioner(use_ocr=True)
```

### HTML Partitioner

The HtmlPartitioner segments and chunks HTML documents by using the embedded structure of the HTML format.

Parameters:

* `skip_headers_and_footers`: Whether to skip headers and footers in the document. Default is `True`.
* `extract_tables`: Whether to extract tables from the HTML document. Default is `False`.
* `text_chunker`: The text chunking strategy to use for processing text content. The default is the `TextOverlapChunker`, and [more info is here](https://sycamore.readthedocs.io/en/model_docs/APIs/data_preparation/functions.html#sycamore.functions.TextOverlapChunker). Default values are: `chunk_token_count: 1000`, `chunk_overlap_token_count: 100`.
* `tokenizer`: The tokenizer to use for tokenizing text content. By default, the 'CharacterTokenizer` is used.

Here is an example of chunking and using table extraction:

```Python
ctx = sycamore.init()
docset = ctx.read.binary(s3://my-bucket/my-folder/, binary_format="html")
            .partition(partitioner=Html_Partitioner(extract_tables=True)
```

Here is an example of chunking and adjusting the chunking strategy:

```Python
ctx = sycamore.init()
docset = ctx.read.binary(s3://my-bucket/my-folder/, binary_format="html")
            .partition(text_chunker=TokenOverlapChunker(chunk_token_count=800, chunk_overlap_token_count=150))
```
