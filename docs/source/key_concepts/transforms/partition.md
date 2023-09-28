## Partition
The Partition transform segments documents into elements. For example, a typical partitioner might chunk a document into elements corresponding to paragraphs, images, and tables. Partitioners are format specific, so for instance for HTML you can use the ``HtmlPartitioner`` and for PDFs, we provide the ``UnstructuredPdfPartitioner``, which utilizes the unstructured open-source library. 

```python
 partitioned_docset = docset.partition(partitioner=UnstructuredPdfPartitioner())
```