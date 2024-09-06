# Process and load data into an OpenSearch hybrid search index

This tutorial provides a walkthrough of how to use Sycamore to extract, enrich, transform, and create vector embeddings from a PDF dataset in S3 and load it into OpenSearch. The way in which you run ETL on these document is critical for the end quality of your application, and you can easily use Sycamore to facilitate this. The example below shows a few transforms Sycamore can do in a pipeline, and how to use LLMs to extract information.

In this example, we will be using PDF documents from the [Sort Benchmark](http://sortbenchmark.org/) website stored in a public S3 bucket. These documents are research papers that contain images, tables, text, and complex formatting.

## Steps

1. Install Sycamore using pip using [these instructions](/sycamore/get_started)

2. Create a Python script and import Sycamore. In the following code snippet, we are initializing Sycamore and creating a DocSet by reading all the files from a local path.

```python
import sycamore

# local file path to the SortBenchmark dataset
paths = "s3://aryn-public/sort-benchmark/pdf/"

# Initializng sycamore which also initializes Ray underneath
context = sycamore.init()

# Creating a DocSet
docset = context.read.binary(paths, parallelism=1, binary_format="pdf")
```

```{note}
At any point if you want to inspect the docset, you can use docset.show() method
```

3. Next, we want to partition all the PDFs and generate elements so that we can extract relevant entities later on. We will use the partition transform to achieve this.

```python
from sycamore.transforms.partition import ArynPartitioner

# We are using Aryn Partitioner to partion the PDFs. By default, it uses the Aryn Partitioning Service. You can sign up for free at https://www.aryn.ai/get-started and set your API key. Set your ARYN_API_KEY variable in your environment variables or in your code for use with the Partition transform.

```
docset = docset.partition(partitioner=ArynPartitioner())
```

4. Now, since we know that titles and authors are important entities in our dataset, let's extract them using OpenAI with the `extract_entity` transform. In this case, we are going to use few shot entity extraction, where we provide some examples to the model of what to extract in our prompt:

```python
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import OpenAIModels, OpenAI
import os

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

# Replace the "api_key" with your API Key.
openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value, api_key=os.environ.get("OPENAI_API_KEY"))

docset = docset.extract_entity(
    entity_extractor=OpenAIEntityExtractor("title", llm=openai_llm, prompt_template=title_prompt_template)
)
.extract_entity(
    entity_extractor=OpenAIEntityExtractor("authors", llm=openai_llm, prompt_template=author_prompt_template)
)
```

5. Next, we want to convert each element of a document into a top/parent level document. Additionally, we want to create embeddings for these documents. We will use the explode and embed transforms respectively to achieve this.

```python
from sycamore.transforms.embed import SentenceTransformerEmbedder

# We are using SentenceTransformerEmbedder to embed the content of each document; which
# uses the SentenceTransformer model. You can write your own Embedder as well.
docset = docset.explode()
.embed(embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
```

6. Lastly, we want to write the data to a hybrid search index (vector and keyword) in OpenSearch using the OpenSearch writer. Make sure the host location and port is correct for your OpenSearch engine. You will need to configure and run OpenSearch separately. 

You can also [write to other target data stores](../sycamore/connectors.html).

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
                        "method": {"name": "hnsw", "engine": "faiss"},
                    },
                    "text": {"type": "text"},
                }
            },
        }
    }

docset.write.opensearch(
        os_client_args=openSearch_client_args,
        index_name="sort-benchmark",
        index_settings=index_settings,
    )
```

Congrats - you have now processed and loaded your PDFs from S3 into an OpenSearch hybrid search index!
