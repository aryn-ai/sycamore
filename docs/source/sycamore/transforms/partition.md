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

* ```use_partitioning_service```: If ```True```, the partitioner uses the *Aryn  Partitioning Service*. Defaults to ```True```.
* ```model_name_or_path```: The HuggingFace coordinates or model local path. It defaults to ```SYCAMORE_DETR_MODEL```, and you should only change it if you are testing a custom model. Ignored when ```use_partitioning_service``` is ```True```.
* ``threshold``: This represents the threshold for accepting the model’s predicted bounding boxes. It defaults to “auto”, where the service uses a processing method to find the best prediction for each possible bounding box. This is the recommended setting. However, you can override this by specifying a numerical threshold between 0 and 1. If you specify a numerical threshold, only bounding boxes with confidence scores higher than the threshold will be returned (instead of using the processing method described above). A lower value will include more objects, but may have overlaps, while a higher value will reduce the number of overlaps, but may miss legitimate objects. If you do set the threshold manually, we recommend starting with a value of 0.32.
* ```use_ocr```: If ```True```, the partitioner uses OCR to extract text from the PDF. It defaults to ```False```, where the partitioner attempts to directly extract the text from the underlying PDF in the bounding box. It currently uses Tesseract for extraction.
* `extract_table_structure`: If `True`, the partitioner runs a separate table extraction model to extract cells from regions of the document identified as tables.
* `table_structure_extractor`: The table extraction implementation to use when `extract_table_structure` is `True`. The default is the `TableTransformerStructureExtractor`. Ignored when ```use_partitioning_service``` is ```True```.
* `extract_images`: If `True`, the partitioner crops each region identified as an image and attaches it to the associated `ImageElement`. This can later be fed into the `SummarizeImages` transform.

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
