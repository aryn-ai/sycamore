## Partition
To process raw documents and bring them into DocSets, Sycamore must first segment the document and label each element, such as headings, tables, and figures. This process is called document partitioning, and it is carried out by the Partition transform. Partitioners are format specific, and we reccomend:

* PDF: Sycamore Partitioner
* HTML: Html Partitioner

Sycamore also includes the ``UnstructuredPdfPartitioner`` for PDFs as well.

```python
 partitioned_docset = docset.partition(partitioner=SycamorePartitioner())
```

### Sycamore Partitioner

The




Partitioner transforms can use a variety of techniques, from hardcoded heuristics to advanced AI models, to identify, classify, and segment unstructured data, and there are many options. With Sycamore, users can choose their partitioner and respective configuration, and we initially included the Unstructured partitioner libraries as part of the stack. However, for real world use cases, we quickly found that this partitioner lacked the fidelity and accuracy we needed for labeling and segmenting data. So, we built our own partitioner utilizing new AI models we’ve created.

We are excited to introduce the Sycamore Partitioner, and it is included in the latest Sycamore release. The first version is focused on PDF documents, and includes a newly trained object detection model that provides better accuracy in labelling and segmenting data. In this blog post, we’ll share more details about the Sycamore Partitioner and examples for how to use it.

The Partition transform segments documents into elements. For example, a typical partitioner might chunk a document into elements corresponding to paragraphs, images, and tables. Partitioners are format specific, so for instance for HTML you can use the ``HtmlPartitioner`` and for PDFs, we provide the ``UnstructuredPdfPartitioner``, which utilizes the unstructured open-source library.

```python
 partitioned_docset = docset.partition(partitioner=UnstructuredPdfPartitioner())
```


The SycamorePartitioner uses an object recognition model to partition the document into
    structured elements.
    Args:
        model_name_or_path: The HuggingFace coordinates or model local path. Should be set to
             the default SYCAMORE_DETR_MODEL unless you are testing a custom model. 
        threshold: The threshold to use for accepting the models predicted bounding boxes. A lower
             value will include more objects, but may have overlaps, a higher value will reduce the
             number of overlaps, but may miss legitimate objects. 
        use_ocr: Whether to use OCR to extract text from the PDF. If false, we will attempt to extract
             the text from the underlying PDF. 
        ocr_images: If set with use_ocr, will attempt to OCR regions of the document identified as images. 
        ocr_tables: If set with use_ocr, will attempt to OCR regions on the document identified as tables.
             Should not be set when `extract_table_structure` is true. 
        extract_table_structure: If true, runs a separate table extraction model to extract cells from
             regions of the document identified as tables. 
        table_structure_extractor: The table extraction implementaion to use when extract_table_structure
             is True. The default is the TableTransformerStructureExtractor. 
        extract_images: If true, crops each region identified as an image and attaches it to the associated
             ImageElement. This can later be fed into the SummarizeImages transform.
    
    Example:
         The following shows an example of using the SycamorePartitioner to partition a PDF and extract
         both table structure and image
    
         .. code-block:: python
            context = scyamore.init()
            partitioner = SycamorePartitioner(extract_table_structure=True, extract_images=True)
            context.read.binary(paths, binary_format="pdf")\
                 .partition(partitioner=partitioner)
    """
