.. image:: images/sycamore_logo.svg
   :alt: Sycamore
====================================

Welcome to Sycamore!
--------------------

Sycamore is an open-source conversational search and analytics platform for complex unstructured data, such as documents, presentations, transcripts, embedded tables, and internal knowledge repositories. It retrieves and synthesizes high-quality answers through bringing AI to data preparation, indexing, and retrieval. Sycamore makes it easy to prepare unstructured data for search and analytics, providing a toolkit for data cleaning, information extraction, enrichment, summarization, and generation of vector embeddings that encapsulate the semantics of data. Sycamore uses your choice of generative AI models to make these operations simple and effective, and it enables quick experimentation and iteration. Additionally, Sycamore uses OpenSearch for storage and queries of vector embeddings and associated data.   



.. image:: images/SycamoreDiagram2.png

**Key Features**

* **Answer Hard Questions on Complex Data.** Prepares and enriches complex unstructured data for search and analytics through advanced data segmentation, LLM-powered UDFs for data enrichment, performant data manipulation with Python, and vector embeddings using a variety of AI models. 

* **Multiple Query Options.** Flexible query operations over unstructured data including RAG, hybrid search, analytical functions, natural language/conversational search, and custom post-processing functions.  

* **Secure and Scalable.** Sycamore leverages OpenSearch, an open-source enterprise-scale search and analytics engine for indexing, enabling hybrid (vector + keyword) search, analytical functions, conversational memory, and more. Also, it offers features like fine-grained access control. OpenSearch is used by thousands of enterprise customers for mission-critical workloads. 

* **Develop Quickly.** Helpful features like automatic data crawlers (Amazon S3 and HTTP) and Jupyter notebook support to create, iterate, and test custom data preparation code. 

* **Plug-and-Play LLMs.** Use different LLMs for entity extraction, vector embedding, RAG, and post-processing steps. Currently supporting OpenAI and Amazon Bedrock, and more to come! 

ADD DEMO VIDEO!!!

Getting Started
--------------------

You can easily deploy Sycamore locally or on a virtual machine using Docker.  

With Docker installed: 

 

1. Clone the Sycamore repo: 

``git clone https://github.com/aryn-ai/sycamore``

2. Set OpenAI Key: 

``export OPENAI_API_KEY=YOUR-KEY``

3. Go to: 

``/sycamore``

4. Launch Sycamore. Containers will be pulled from DockerHub: 

``docker compose up --pull=always``


**Use the service:**

a. Load demo dataset via (NEED LINK tutorial here): 

``docker compose run crawl_sort_benchmark``

b. Load custom data via (NEED LINK tutorial here): 

``docker compose run crawl_http http://my.website.example.com``

c. Write custom data ingestion and preparation code using the Jupyter container (NEED LINK tutorial here). Access it via the URL from: 

``docker compose logs jupyter | grep Visit``


Once you've loaded data, you can run conversational search on your data with the Sycamore demo UI at localhost:3000  
 

More Resources
--------------------
- Join the Sycamore Slack workspace: Link 
- View the Sycamore GitHub 
- To learn more about Sycamore’s architecture, click here NEED LINK
- For more info about data preparation in Sycamore, visit here [link to docs] 
- To integrate your own query front-end to Sycamore, visit here  NEED LINK




.. toctree::
   :caption: Key Concepts
   :maxdepth: 2
   :hidden:

   ../key_concepts/concepts.md
   ../key_concepts/transforms.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   ../tutorials/end_to_end_tutorials.md

.. toctree::
   :caption: APIS
   :maxdepth: 2
   :hidden:

   ../APIs/index.rst
