These instructions have been tested on MacOS 14.0, MacOS 13.4.1, and Ubuntu 23.10; they may have to be tweaked for other environments

# Write and iterate on Sycamore jobs locally

The Quickstart configuration for Aryn Search easily launches containers for the full stack. However, you may prefer to write and iterate on your Sycamore data processing scripts locally, and load the output of these tests into the containerized Aryn stack. A benefit of this approach is that you can use a local Jupyter Notebook to develop your scripts.

In this example, we will:

- [Install and run components](#Install-and-run-components)
- [Write an initial Sycamore job](#Write-an-initial-Sycamore-job)
- [Add metadata extraction using GenAI](#Add-metadata-extraction-using-GenAI)

The full notebook that includes the final code of the Sycamore job is [here](https://github.com/aryn-ai/quickstart/blob/main/sycamore_local_dev_example.ipynb).

## Install and run components

1. Launch Aryn Search using the containerized Quickstart following [these instructions](https://github.com/aryn-ai/quickstart#readme). However, a few notes on this step specific to this example:

- This example doesn't need Amazon Textract or Amazon S3, so you do not need to have or provide AWS credentials.
- You do not need to load the full Sort Benchmark sample dataset referred to in the Quickstart README.

2. Install [Sycamore](https://github.com/aryn-ai/sycamore) locally.


```
python3 -m venv .
. bin/activate
```


```
pip install sycamore-ai
```

For certain PDF processing operations, you also need to install poppler, which you can do with the OS-native package manager of your choice. 

For example, the command for Homebrew on Mac OS is:

```
brew install poppler
```

For Linux:
```
sudo apt install poppler-utils
```

3. Install [Jupyter Notebook](https://jupyter.org/). If you already have a python notebook environment, you can choose to use that instead.

```
pip install notebook
```


## Write an initial Sycamore job

1. Download the PDFs we want to use for our sample dataset. We will use two journal articles, "Attention Is All You Need" and "A Comprehensive Survey On Applications Of Transformers For Deep Learning Tasks." We will download these locally:

```
(
    mkdir -p tmp/sycamore/data
    curl -O https://arxiv.org/pdf/1706.03762.pdf --output-dir tmp/sycamore/data
    curl -O https://arxiv.org/pdf/2306.07303.pdf --output-dir tmp/sycamore/data
)
```

2. Launch Jupyter Notebook

If you haven't set your OpenAI Key, do so before starting your notebook:
```
export OPENAI_API_KEY=YOUR-KEY
```

Before you start up the notebook, make sure OpenSearch is running.  

```
curl localhost:9200
```

If it's not, then start up the container using the [Quickstart instructions](https://github.com/aryn-ai/quickstart).

If the OpenSearch is running (and you get a response), then run:
```
jupyter notebook
```

In the browser window that appears, navigate to File -> New -> Notebook; and then choose the
preferred kernel.

3. Write initial Sycamore Job.

3a. First, we will import our dependencies from IPython, JSON, Pillow, and Sycamore:

```python
import json
import os
import sys


from IPython.display import display, Image
from IPython.display import IFrame
from PIL import Image as PImage

import sycamore
from sycamore.data import Document
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import OpenAIModels, OpenAI, LLM
from sycamore.transforms.partition import UnstructuredPdfPartitioner, HtmlPartitioner
from sycamore.llms.prompts.default_prompts import TEXT_SUMMARIZER_GUIDANCE_PROMPT_CHAT
from sycamore.transforms.summarize import Summarizer
from sycamore.transforms.extract_table import TextractTableExtractor
from sycamore.functions.document import split_and_convert_to_image, DrawBoxes
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.scans.file_scan import JsonManifestMetadataProvider
```

3b. Next, we will include the creation of a metadata file that enables our demo UI to show and highlight the source documents when clicking on a search result. In this example, the demo UI will pull the document from a publicly accessible URL. However, you could choose to host the documents in Amazon S3 (common for enterprise data) or other locations accessible by the demo UI container.

You can make additional cells with the rectangle over a + button on the right side of each cell.

```python

metadata = {
  "tmp/sycamore/data/1706.03762.pdf" : {
    "_location": "https://arxiv.org/pdf/1706.03762.pdf"
  }, "tmp/sycamore/data/2306.07303.pdf" : {
    "_location": "https://arxiv.org/pdf/2306.07303.pdf"
  }
}

with open("tmp/sycamore/manifest.json", "w") as f:
    json.dump(metadata, f)
manifest_path = "tmp/sycamore/manifest.json"
```

3c. The next two cells will show a quick view of the PDF documents we will ingest, if we want to inspect them or take a closer look:

```
IFrame(str("tmp/sycamore/data/1706.03762.pdf"), width=700, height=600)
```

```
IFrame(str("tmp/sycamore/data/2306.07303.pdf"), width=700, height=600)
```

3d. Next, we will set some variables:

```python
paths = "tmp/sycamore/data/"

openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
```
If you are using Linux, you may not have the Arial font installed. If so, choose a different font.

3e. Now, we initialize Sycamore, and create a [DocSet](https://sycamore.readthedocs.io/en/stable/key_concepts/concepts.html):

```python
context = sycamore.init()
pdf_docset = context.read.binary(paths, binary_format="pdf", metadata_provider=JsonManifestMetadataProvider(manifest_path))

pdf_docset.show(show_binary = False)
```

If you are using Linux, you may need to install this package:

```
pip install async_timeout
```

The output of this cell will show information about the DocSet, and show that there are two documents included in it.

3f. This cell will segment the PDFs and visually show how a few pages are segmented. 

```python
# Note: these fonts aren't correct, but are close enough for the visualization
if os.path.isfile("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
    font_path = "LiberationSans-Regular"
else:
    print("Using default Arial font, which should work on MacOS and Windows")
    font_path = "Arial.ttf"

def filter_func(doc: Document) -> bool:
    return doc.properties["page_number"] == 1

partitioned_docset = pdf_docset.partition(partitioner=UnstructuredPdfPartitioner())
visualized_docset = (partitioned_docset
              .flat_map(split_and_convert_to_image)
              .map_batch(DrawBoxes, f_constructor_args=[font_path])
              .filter(filter_func))

for doc in visualized_docset.take(2):
    display(Image(doc.binary_representation, height=500, width=500))
```

3g. Next, we will merge the initial chunks from the document segmentation into larger chunks. We will set the maximum token size so the larger chunk will still fit in the context window of our transformer model, which we will use to create vector embeddings in a later step. We have seen larger chunk sizes improve search relevance, as the larger chunk gives more contextual information about the data in the chunk to the transformer model.

```python
merged_docset = partitioned_docset.merge(GreedyTextElementMerger(tokenizer=HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2"), max_tokens=512))
merged_docset.show(show_binary = False)
```

The output should show a dict with an array of elements, and each element with content like `'type': 'Section', 'binary_representation': b'...'`.

3h. Now, we will explode the DocSet and prepare it for creating vector embeddings and loading into OpenSearch. The explode transform converts the elements of each document into top-level documents.

```python
exploded_docset = merged_docset.explode()
exploded_docset.show(show_binary = False)
```

The output should show the exploded DocSet with an empty 'elements' list in the `‘type’: ‘pdf’` entry, many entries with `‘type’: ‘Section’`, and no 'elements' sub-entry.

3i. We will now create the vector embeddings for our DocSet. The model we selected is MiniLM, and you could choose a different embedding model depending on your use case.


```python
st_embed_docset = (exploded_docset
              .embed(embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")))
st_embed_docset.show(show_binary = False)
```

The output should show the DocSet with vector embeddings, e.g. `'embedding': '<384 floats>'` should be added to each of the sections.

3j. Before loading the OpenSearch component of Aryn Search, we need to configure the Sycamore job to: 1/communicate with the Aryn OpenSearch container and 2/have the proper configuration for the vector and keyword indexes for hybrid search. Sycamore will then create and load those indexes in the final step.

The rest endpoint for the Aryn OpenSearch container from the Quickstart is at localhost:9200.  Make sure to provide the name for the index you will create. OpenSearch is a enterprise-grade, customizable search engine and vector database, and you can adjust these settings depending on your use case.


```python
index = "local_development_example_index" # You can change this to something else if you'd like

os_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": False,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

index_settings =  {
        "body": {
            "settings": {"index.knn": True, "number_of_shards": 5, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "dimension": 384,
                        "method": {"engine": "nmslib", "space_type": "l2", "name": "hnsw", "parameters": {}},
                        "type": "knn_vector",
                    },
                    "title": {"type": "text"},
                    "searchable_text": {"type": "text"},
                    "title_embedding": {
                        "dimension": 384,
                        "method": {"engine": "nmslib", "space_type": "l2", "name": "hnsw", "parameters": {}},
                        "type": "knn_vector",
                    },
                    "url": {"type": "text"},
                }
            },
        }
    }
```

3k. This is the final part of the Sycamore job. We will load the data and vector embeddings into the OpenSearch container using the configuration supplied above.

```python
st_embed_docset.write.opensearch(os_client_args=os_client_args, index_name=index, index_settings=index_settings)
```

3l. Finally, add a cell to tell you where to go next and which index you should be querying.  This cell is also useful when you re-execute the script as the output appears when all cells are done.

```python
print("Visit http://localhost:3000 and use the", index, " index to query these results in the UI")
```

4. Once the data is loaded into OpenSearch, you can use the demo UI for conversational search on it.
- Using your internet browser, visit http://localhost:3000 . Make sure the demo UI container is still running from the Quickstart.
- Make sure the index selected in the dropdown has the same name you provided in step 3j
- Create a new conversation. Enter the name for your conversation in the text box in the left "Conversations" panel, and hit enter or click the "add convo" icon on the right of the text box.
- As a sample question, you can ask "Who wrote Attention Is All You Need?"

The results of the hybrid search are in the right hand panel, and you can click through to find the highlighted passage (step 3b enabled this). Though we are getting good results back from hybrid search, it would be nice if we could have the titles and other information for each passage. In the next section, we will iterate on our Sycamore job, and use generative AI to extract some metadata.

## Add metadata extraction using GenAI

Now we are going to edit our existing notebook to show how you could adjust the Sycamore
processing.  Because we are editing the notebook, we will be making changes to the cells partway
through the notebook, and then editing some of the nearby cells to connect the new processing into
the old processing.

5a. Scroll back in the notebook and add a cell between the `visualized_docset` cell and before the `merged_docset` cell. The new cell is between the ones you added in step 3f and 3g.

5b. In this cell, we will add prompt templates for extracting titles and authors. These prompts train a generative AI model to identify a title (or author) by giving examples, and then we will use the trained model to identify and extract them for each document.

```python
 title_context_template = """
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
author_context_template = """
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
```

5c. Add a following cell. In this cell, we will use Sycamore's entity extractor with the prompt templates. We are selecting OpenAI as the generative AI model to use for this extraction.

```python
entity_docset = (partitioned_docset
                 .extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=openai_llm, prompt_template=title_context_template))
                 .extract_entity(entity_extractor=OpenAIEntityExtractor("authors", llm=openai_llm, prompt_template=author_context_template)))

entity_docset = entity_docset.spread_properties(["title", "authors"])

entity_docset.show(show_binary = False, show_elements=False)
```

The output should show the title and author added to the elements in the DocSet.

5d. Since we have changed the output of this cell, we need to use the entity_docset rather than the
partitioned_docset to create the merged_docset.  Adjust the next cell so that it looks like:

```python
merged_docset = entity_docset.merge(GreedyTextElementMerger(tokenizer=HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2"), max_tokens=512))
merged_docset.show(show_binary = False)
```

5e. Change the index name set below (e.g. to `index = "local_development_example_index_withentity"`) that you added in step 3j so that when you run the remaining cells it will load into a new index. Otherwise the old and new data processed data would be intermingled.

5f. Run the rest of the cells in the notebook, and load the data into OpenSearch.

5g. Once the data is loaded into OpenSearch, you can use the demo UI for conversational search on it.
- Using your internet browser, visit http://localhost:3000 . Make sure the demo UI container is still running from the Quickstart
- Make sure the index selected in the dropdown has the same name you provided in the previous step
- The titles should appear with the hybrid search results in the right panel. If they don't check that you both a) changed the index name, and b) used the new index in the UI.

Congrats! You've developed and iterated on a Sycamore data preparation script locally, and used generative AI to extract metadata and enrich your dataset. As your datatset changes, you could automate this processing job using the Sycamore container deployed in the Quickstart configuration.
