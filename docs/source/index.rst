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


Getting Started
--------------------

You can easily deploy Sycamore locally or on a virtual machine using Docker.

With Docker installed:

1.	Clone the Sycamore repo:

``git clone https://github.com/aryn-ai/sycamore``

2.	Set OpenAI Key:

``export OPENAI_API_KEY=YOUR-KEY``

3.	Go to:

``/sycamore``

4.	Launch Sycamore. Containers will be pulled from DockerHub:

``docker compose up --pull=always``

5.	The Sycamore demo query UI is located at:

``http://localhost:3000/``

You can next choose to run a demo that :doc:`prepares and ingests data from the Sort Benchmark website </welcome_to_sycamore/get_started.md#demo-ingest-and-query-sort-benchmark-dataset>`, :doc:`crawl data from a public website <welcome_to_sycamore/get_started.md#demo-ingest-and-query-data-from-an-arbitrary-website>`, or :doc: `write your own data preparation script </data_ingestion_and_preparation/using_jupyter>`.

For more info about Sycamore’s data ingestion and preparation feature set, visit the :doc:`Sycamore documentation </data_ingestion_and_preparation/data_preparation_concepts>`.


Run a demo
--------------------

a. Load demo dataset using the HTTP crawler :doc:`as shown in this tutorial </welcome_to_sycamore/get_started.md#demo-ingest-and-query-sort-benchmark-dataset>`:

``docker compose run crawl_sort_benchmark``

b. Load website data via HTTP crawler :doc:`as shown in this tutorial </welcome_to_sycamore/get_started.md#demo-ingest-and-query-data-from-an-arbitrary-website>`:

``docker compose run crawl_http http://my.website.example.com``

c. Write :doc:`custom data ingestion and preparation code using the Jupyter container </data_ingestion_and_preparation/using_jupyter>`. Access it via the URL from:

``docker compose logs jupyter | grep Visit``

Once you've loaded data, you can run conversational search on your data with the Sycamore demo query UI at localhost:3000

For more details about getting started, visit the :doc:`Sycamore Getting Started page </welcome_to_sycamore/get_started>`.

More Resources
--------------------
- Join the Sycamore Slack workspace: https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg
- View the Sycamore GitHub: https://github.com/aryn-ai/sycamore
- :doc:`Learn more about Sycamore’s architecture </welcome_to_sycamore/architecture>`
- :doc:`Learn more about data preparation in Sycamore <data_ingestion_and_preparation/data_preparation_concepts>`
- :doc:`Integrate your query app with Sycamore </querying_data/integrate_your_application>`


.. toctree::
   :caption: Welcome to Sycamore
   :maxdepth: 2
   :hidden:

   /welcome_to_sycamore/get_started.md
   /welcome_to_sycamore/architecture.md
   /welcome_to_sycamore/hardware.md
   /welcome_to_sycamore/encryption.md


.. toctree::
   :caption: Data Ingestion and Preparation
   :maxdepth: 2
   :hidden:

   /data_ingestion_and_preparation/data_preparation_concepts.md
   /data_ingestion_and_preparation/load_data.md
   /data_ingestion_and_preparation/using_jupyter.md
   /data_ingestion_and_preparation/installing_sycamore_libraries_locally.md
   /data_ingestion_and_preparation/running_a_data_preparation_job.md
   /data_ingestion_and_preparation/generative_ai_configuration.md
   /data_ingestion_and_preparation/transforms.rst
   /data_ingestion_and_preparation/connectors.rst


.. toctree::
   :caption: Querying Data
   :maxdepth: 2
   :hidden:

   /querying_data/demo_query_ui.md
   /querying_data/using_rag_pipelines.md
   /querying_data/hybrid_search.md
   /querying_data/reranking.md
   /querying_data/remote_processors.md
   /querying_data/dedup.md
   /querying_data/integrate_your_application.md
   /querying_data/generative_ai_configurations.md



.. toctree::
   :caption: Conversation Memory
   :maxdepth: 2
   :hidden:


   /conversation_memory/overview.md
   /conversation_memory/storage_for_genai_agents.md
   /conversation_memory/using_with_conversational_search.md


.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   /tutorials/sycamore-jupyter-dev-example.md
   /tutorials/sycamore_data_prep_local.md
   /tutorials/conversational_memory_with_langchain.md


.. toctree::
   :caption: APIs
   :maxdepth: 2
   :hidden:

   /APIs/data_preparation.rst
   /APIs/conversation_memory.rst
   /APIs/transforms.rst
