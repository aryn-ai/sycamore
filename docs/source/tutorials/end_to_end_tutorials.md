# Read from s3 and write to OpenSearch


We strongly encourage to go over Key Concepts[Link] before you work through this tutorial. 

This tutorial gives a walk through over how you can leverage Sycamore to prepare, enhance and embed your data and eventually write it to a local OpenSearch cluster. We will be using unstructured data from the Sort Benchmark (sortbenchmark.org) website, including HTML webpages and PDF documents that include tables and figures for this tutorial. The data is available at s3Location for you to download. 

We will be using OpenSearch as a VectorStore. Please install and spin up OpenSearch locally using Docker by following the instructions [here](https://opensearch.org/docs/latest/install-and-configure/install-opensearch/docker/).

1. Install Sycamore using Pypi

```bash
pip install sycamore-ai
```

1. Create a python script and import Sycamore. In the following code snippet, we are initializing sycamore and creating a DocSet by reading all the files from a local path. 

```python
import sycamore

# local file path to the SortBenchmark dataset
paths = "data/pdfs/"

# Initializng sycamore which also initializes Ray underneath
context = sycamore.init()

# Creating a DocSet 
docset = context.context.read.binary(paths, binary_format="pdf")
```

*Note: At any point if you want to inspect the docset, you can use docset.show() method*

1. Next, we want to partition all the pdfs and generate elements so that we can extract relevant entities later on. We will use the partition transform to achieve this.

```python
from sycamore.execution.transforms.partition import UnstructuredPdfPartitioner

# We are using UnstructuredPdfPartitioner to partion the documents.
# You can write your ownn partitioner as well. 
docset = docset.partition(partitioner=UnstructuredPdfPartitioner())
```

1. Now, since we know that Titles and Authors are important entities of our dataset, lets extract them using the extract_entity transform. The extracted entities will be added as properties of each document.

```python
from sycamore.execution.transforms.entity_extraction import OpenAIEntityExtractor
from sycamore.execution.transforms.llms.llms import OpenAIModels, OpenAI

# The following prompt templates will be used to extract the relevant entities 
title_prompt_template = """
    ELEMENT 1: Jupiter's Moons
    ELEMENT 2: Ganymede 2020
    ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
    ELEMENT 4: From Wikipedia, the free encyclopedia
    ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
    =========
    "Ganymede 2020"

    ELEMENT 1: FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
    ELEMENT 2: Tarun Kalluri * UCSD
    ELEMENT 3: Deepak Pathak CMU
    ELEMENT 4: Manmohan Chandraker UCSD
    ELEMENT 5: Du Tran Facebook AI
    ELEMENT 6: https://tarun005.github.io/FLAVR/
    ELEMENT 7: 2 2 0 2
    ELEMENT 8: b e F 4 2
    ELEMENT 9: ]
    ELEMENT 10: V C . s c [
    ========
    "FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation"

    """

author_prompt_template = """
    ELEMENT 1: Jupiter's Moons
    ELEMENT 2: Ganymede 2020
    ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
    ELEMENT 4: From Wikipedia, the free encyclopedia
    ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
    =========
    Audi Laupe, Serena K. Goldberg

    ELEMENT 1: FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
    ELEMENT 2: Tarun Kalluri * UCSD
    ELEMENT 3: Deepak Pathak CMU
    ELEMENT 4: Manmohan Chandraker UCSD
    ELEMENT 5: Du Tran Facebook AI
    ELEMENT 6: https://tarun005.github.io/FLAVR/
    ELEMENT 7: 2 2 0 2
    ELEMENT 8: b e F 4 2
    ELEMENT 9: ]
    ELEMENT 10: V C . s c [
    ========
    Tarun Kalluri, Deepak Pathak, Manmohan Chandraker, Du Tran

    """

# We are using OpenAIEntityExtractor which utilizes OpenAI and gpt-3.5-turbo model.
# You can write your own EntityExtractor as well.

# Replace the "api-key" with your API Key.
openai = OpenAI(OpenAIModels.GPT_3_5_TURBO.value, "api-key")

docset = docset.extract_entity(
            entity_extractor=OpenAIEntityExtractor("title", llm=openai_llm, prompt_template=title_prompt_template)
        )
        .extract_entity(
            entity_extractor=OpenAIEntityExtractor("authors", llm=openai_llm, prompt_template=author_prompt_template)
        )
```

1. Next, we want to convert each element of a document into a top/parent level document. Additionally, we also want to generate embeddings for these documents. We will use the explode and embed transform respectively to achieve this.

```python
from sycamore.execution.transforms.embedding import SentenceTransformerEmbedder

# We are using SentenceTransformerEmbedder to embed the content of each document; which
# uses the SentenceTransformer model. You can write your own Embedder as well.
docset = docset.explode()
		.embed(embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
```

1. Lastly, we want to write these documents into OpenSearch to query. Make sure that you have OpenSearch running locally.

```python
openSearch_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    index_settings = {
        "body": {
            "settings": {
                "index.knn": True,
                "number_of_shards": 5,
                "number_of_replicas": 1,
            },
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {"name": "hnsw", "engine": "nmslib"},
                    },
                    "text": {"type": "text"},
                }
            },
        }
    }

docset.write.opensearch(
        os_client_args=openSearch_client_args,
        index_name="sycamoreindex",
        index_settings=index_settings,
    )
```