# Data Preparation Concepts

You can use the [default data preparation code](../../../notebooks/default-prep-script.ipynb) to segment, process, enrich, embed, and load your data into Sycamore. This runs automatically when using the [crawlers to load data](..//load_data.md#using-a-crawler), and is used in the [Get Started examples](../welcome_to_sycamore/get_started.md). However, to get the best results on complex data, you will likely need to write custom code specific for your data to prepare it for search and analytics.

Sycamore provides a toolkit for data cleaning, information extraction, enrichment, summarization, and generation of vector embeddings that encapsulate the semantics of your data. It uses your choice of generative AI models to make these operations simple and effective, and it enables quick experimentation and iteration. You write your data preparation code in Python, and Sycamore uses Ray to easily scale as your workloads grow.

Sycamore data preparation code uses the concepts below, and available transforms are [here](/transforms.rst). Also, as an example, you can view the code for the default data preparation code [here](https://github.com/aryn-ai/sycamore/blob/main/notebooks/default-prep-script.ipynb) and learn more about how to run your custom code [here](/running_a_data_preparation_job.md).

## Sycamore data preparation concepts

### DocSet

A DocSet, short for “documentation set,” is a distributed collection of documents bundled together for processing. Sycamore provides a variety of transformations on DocSets to help customers handle unstructured data easily. For example, the following code snippet shows several transforms chained together to process a collection of PDF documents.

```context = sycamore.init()
docset = context.read\
    .binary("s3://bucket/prefix", binary_format="pdf")\
    .partition(partitioner=UnstructuredPdfPartitioner())\
    .explode()\
    .sketch()\
    .embed(SentenceTransformerEmbedder(
        batch_size=10000, model_batch_size=1000,
        model_name="sentence-transformers/all-MiniLM-L6-v2"))
```


### Document

A Document is a generic representation of an unstructured document in a format like PDF or HTML. Though different types of Documents may have different properties, they all contain [the following common fields](https://sycamore.readthedocs.io/en/stable/APIs/data/data.html#sycamore.data.document.Document):

* **binary_representation:** The raw content of the document. May not be present in elements after partitioning of non-binary inputs such as HTML.

* **doc_id:** A unique identifier for the Document. Defaults to a UUID.

* **elements:** A list of elements belonging to this Document. If the document has no elements, for example before it is chunked, this field will be [].

* **embedding:** The embedding associated with the document (usually after it is partitioned) or None if it hasn't been set.

* **parent_id:** In Sycamore, certain operations create parent-child relationships between Documents. For example, the explode transform promotes elements to be top-level Documents, and these Documents retain a pointer to the Document from which they were created using the parent_id field. For those Documents which have no parent, parent_id is None.

* **properties:** A dictionary of system or customer defined properties. By default a Document will have 'path' and '_location' attributes. Additional processing can add extra attributes such as title or author.

* **text_representation:** The extracted text from the Document; this representation is created in the elements after a .partition() step and promoted to top level Documents after an .explode() step.

* **type:** The type of the Document, e.g. pdf, html.

Documents may have additional, less important fields, see [the code](https://github.com/aryn-ai/sycamore/blob/main/lib/sycamore/sycamore/data/document.py#L8) or [the auto-generated documentation](https://sycamore.readthedocs.io/en/stable/APIs/data_preparation/document.html) for an exhaustive list.


### Element

It is often useful to process different parts of a Document separately. For example, you might want to process tables differently than text paragraphs, and typically small chunks of text are embedded separately for vector search. In Sycamore, these chunks are called Elements. Like Documents, Elements contain text or binary representations and collection of properties that can be set by the user or by built-in transforms.

### Query Execution

In Sycamore, DocSet evaluation is lazy, which means that transforms on DocSet aren’t executed until needed by an operation like show or write. Internally, the transforms are converted to an execution plan in the backend. This lazy execution framework provides opportunities to sanitize and optimize the query execution. For instance, we could convert the above example DocSet transformations into the following execution plan:

![Untitled](imgs/query_execution.svg)
