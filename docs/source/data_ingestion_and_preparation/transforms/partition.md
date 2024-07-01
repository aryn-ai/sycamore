(Partition)=
## Partition
To process raw documents and bring them into DocSets, Sycamore must first segment and chunk the document and label each element, such as headings, tables, and figures. This process is called document partitioning, and it is carried out by the Partition transform. Partitioners are format specific, and we reccomend:

* PDF: Sycamore Partitioner
* HTML: Html Partitioner

Sycamore also includes the ``UnstructuredPdfPartitioner`` for PDFs as well.

```python
 partitioned_docset = docset.partition(partitioner=SycamorePartitioner())
```

### Sycamore Partitioner

The Sycamore Partitioner was built from the ground-up for high-quality segmentation using a new AI vision model at it's core. This model is [a Deformable DEtection Transformer (DETR) model](https://huggingface.co/Aryn/deformable-detr) trained on [DocLayNet](https://github.com/DS4SD/DocLayNet), an open source, human-annotated document layout segmentation dataset. This model is 100% open source with an Apache v2.0 license.

There are several options you can use in the Sycamore Partitioner for table extraction, OCR, and more.

Parameters:

* ```model_name_or_path```: The HuggingFace coordinates or model local path. It defaults to ```SYCAMORE_DETR_MODEL```, and you should only change it if you are testing a custom model. * * ```threshold```: The threshold to use for accepting the models predicted bounding boxes. A lower value will include more objects, but may have overlaps, a higher value will reduce the number of overlaps, but may miss legitimate objects. It defaults to ```0.4```.
* ```use_ocr```: If ```True```, the partitioner uses OCR to extract text from the PDF. It defaults to ```False```, where the partitioner attempts to directly extract the text from the underlying PDF in the bounding box. It currently uses Tesseract for extraction.
* ```ocr_images```: If set to ```True``` alongside ```use_ocr```, the partitioner will attempt to OCR regions of the document identified as images.
* ```ocr_tables```: If set to ```True``` alongside ```use_ocr```, the partitioner will attempt to OCR regions on the document identified as tables. This should not be set when `extract_table_structure` is ```True```. It currently uses EasyOCR for extraction.
* `extract_table_structure`: If `True`, the partitioner runs a separate table extraction model to extract cells from regions of the document identified as tables. Do not set if 'ocr_tables' is true.
* `table_structure_extractor`: The table extraction implementation to use when `extract_table_structure` is `True`. The default is the `TableTransformerStructureExtractor`.
* `extract_images`: If `True`, the partitioner crops each region identified as an image and attaches it to the associated `ImageElement`. This can later be fed into the `SummarizeImages` transform.

Here is an example of chunking and using table extraction:

```Python
ctx = sycamore.init()
docset = ctx.read.binary(s3://my-bucket/my-folder/, binary_format="pdf")
            .partition(partitioner=SycamorePartitioner(extract_table_structure=True))
```

Here is an example of chunking and using OCR, including OCR for tables and images:

```Python
ctx = sycamore.init()
docset = ctx.read.binary(s3://my-bucket/my-folder/, binary_format="pdf")
            .partition(partitioner=SycamorePartitioner(use_ocr=True, ocr_images=True, ocr_tables=True)
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
