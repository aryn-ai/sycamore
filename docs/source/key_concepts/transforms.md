# Transforms

Sycamore has various transforms which operate on the docset to prepare and enhance your unstructured data. Majority of the
transforms are designed such that you can easily pass in your implementation of the transform.

## Embed
The Embed Transform is responsible for generating embeddings for your documents or specific document segments.
These generated embeddings are then incorporated into the document structure under the designated "embedding" property.
Sycamore will automatically batch records and leverage GPUs where appropriate during the embedding generation process.
Furthermore, you have the flexibility to seamlessly integrate various embedding models by implementing the Embedder class.
For example, the SentenceTransformerEmbedder uses SentenceTransformer model to generate embeddings.

```python
embedder = SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
embedded_doc_set = docset.embed(embedder)
```

## Explode
The Explode Transform converts document elements into higher-level parent documents. This is primarily useful when
you want to ingest chunks of your document as independent records into a data store like OpenSearch.

```python
exploded_doc_set = docset.explode()
```
## Extract_Entity
The Extract Entity Transform extracts semantically meaningful information from your documents with just a few examples
by leveraging LLMs. These extracted entities are then incorporated into the document structure under the properties.
You can easily configure this transform by providing your implementation of the EntityExtractor class. For instance, in the following example
the OpenAIEntityExtractor utilizes an OpenAI model gpt-3.5-turbo to extract the entity "title" :

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

docset = docset.extract_entity("title", llm=openai_llm, prompt_template=title_context_template)
```

## FlatMap
As the name suggests, the FlatMap transform operates on each document and outputs a list of document by applying the
user defined function on each document. In the following example, the FlatMap transform outputs a new list of documents
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
The Map transform
```python

```
## MapBatch
The MapBatch transform
```python

```

## Partition
The Partition Transform breaks raw documents into standard, structured chunks or elements. The generated elements are put
into the document under "elements". The partition transform takes in Partitioner to chunk the documents. In the following example,
we are using an open source pdf partitioner from Unstructured.io to chunk the documents.

```python
 partitioned_docset = docset.partition(partitioner=UnstructuredPdfPartitioner())
```
## Summarize
Similar to the Extract Entity transform, Summarize transform helps summarize your documents or part of your documents
leveraging LLMs. Just like other transforms, you can easily configure this transform by providing your implementation of
the Summarizer class. For instance, in the following example, we are using LLMElementTextSummarizer with the summarize transform
to create the summaries of the elements which are longer than a certain length.

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
