# Transforms

In Sycamore, a transform is a method that operates on a ``DocSet`` and returns a new ``DocSet``. Sycamore provides a number of these transforms directly in the ``DocSet`` class to prepare and enhance your unstructured data. In order to support a variety of data types and machine learning models, many of these transforms are customizable with different implementations.

## Embed
The Embed Transform is responsible for generating embeddings for your Documents or Elements. These embeddings are stored in a special ``embedding`` property on each document. 
The initial embedding implementation is the ``SentenceTransformerEmbedder``, which embeds the text representation of each document using any of the models from the popular [SentenceTransformers framework](https://www.sbert.net/). For example, the following code embeds a ``DocSet`` with the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model:

```python
embedder = SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
embedded_doc_set = docset.embed(embedder)
```

During execution, Sycamore will automatically batch records and leverage GPUs where appropriate.

## Explode

The Explode transform converts the elements of each document into top-level documents. For example, if you explode a ``DocSet`` with a single document containing two elements, the resulting ``DocSet`` will have three documents -- the original plus a new ``Document`` for each of the elements.

```python
exploded_doc_set = docset.explode()
```

The primary use of the explode transform is to embed and ingest chunks of your document, the elements, as independent records in a data store like OpenSearch.

## ExtractEntity
The Extract Entity Transform extracts semantically meaningful information from your documents. The ``OpenAIEntityExtractor`` leverages one of OpenAI's LLMs to perform this extraction with just a few examples. These extracted entities are then incorporated as properties into the document structure. The following code shows how to provide an example template for extracting a title using the gpt-3.5-turbo model. 

```python
openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
title_prompt_template = """
    ELEMENT 1: Jupiter's Moons
    ELEMENT 2: Ganymede 2020
    ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
    ELEMENT 4: From Wikipedia, the free encyclopedia
    ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
    =========
    "Ganymede 2020
"""

docset = docset.extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=openai_llm, prompt_template=title_context_template))
```

## FlatMap
The FlatMap transform takes a function from a single ``Document`` to a list of ``Documents``, and returns then "flattens" the result into a single ``DocSet``. In the following example, the FlatMap transform outputs a new list of documents
where each document includes elements from a single page only.
```python
def split_and_convert_to_image(doc: Document) -> list[Document]:
    if doc.binary_representation is not None:
        images = pdf2image.convert_from_bytes(doc.binary_representation)
    else:
        return [doc]

    elements_by_page: dict[int, list[Element]] = {}

    for e in doc.elements:
        page_number = e.properties["page_number"]
        elements_by_page.setdefault(page_number, []).append(e)

    new_docs = []
    for page, elements in elements_by_page.items():
        new_doc = Document(elements={"array": elements})
        new_doc.properties.update(doc.properties)
        new_doc.properties.update({"page_number": page})
        new_docs.append(new_doc)
    return new_docs

docset = docset.flat_map(split_and_convert_to_image)
```

## Map
The Map transform takes a function that takes a ``Document`` and returns a ``Document``, 
and applies it to each document in the ``DocSet``. 


## MapBatch
The MapBatch transform is similar to ``Map``, except that it processes a list of documents and returns a list of documents. ``MapBatches`` is ideal for transformations that get performance benefits from batching. 


## Partition
The Partition transform segments documents into elements. For example, a typical partitioner might chunk a document into elements corresponding to paragraphs, images, and tables. Partitioners are format specific, so for instance for HTML you can use the ``HtmlPartitioner`` and for PDFs, we provide the ``UnstructuredPdfPartitioner``, which utilizes the unstructured open-source library. 

```python
 partitioned_docset = docset.partition(partitioner=UnstructuredPdfPartitioner())
```

## Summarize
Similar to the extract entity transform, the summarize transform generates summaries of documents or elements. The ``LLMElementTextSummarizer`` summarizes a subset of the elements from each Document. It takes an LLM implementation and a callable specifying the subset of elements to summarize. The following examples shows how to use this transform to summarize elements that are longer than a certain length. 

```python
def filter_elements_on_length(
    document: Document,
    minimum_length: int = 10,
) -> list[Element]:
    def filter_func(element: Element):
        if element.text_representation is not None:
            return len(element.text_representation) > minimum_length

    return filter_elements(document, filter_func)

llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)

docset = docset.summarize(LLMElementTextSummarizer(llm, filter_elements_on_length))
```
