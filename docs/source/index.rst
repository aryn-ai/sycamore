Welcome to Aryn!
--------------------
Aryn is an LLM-powered data preparation, processing, and analytics system for complex, unstructured documents like PDFs, HTML, presentations, and more. With Aryn, you can prepare data for GenAI and RAG applications, power high-quality document processing workflows, and run analytics on large document collections with natural language. It includes two components: The Aryn Partitioning Service and Sycamore.

The Aryn Partitioning Service (APS) is a serverless, GPU-powered API for segmenting and labeling PDF documents, doing OCR, and extracting tables and images. It returns the output in JSON. APS runs the Aryn Partitioner and it’s `state-of-the-art, open source deep learning DETR AI model<https://huggingface.co/Aryn/deformable-detr-DocLayNet>`_ trained on 80k+ enterprise documents. You can use it to partition documents and extract information directly in your code, or use with Sycamore for additional processing. `Sign-up here for free<https://www.aryn.ai/get-started>`_ to get an API Key and use the `Aryn Playground<https://play.aryn.cloud/partitioning>`_ to visually see how it segments and processes your own documents. Or, watch the Aryn Partitioning Service in action in `this video<https://www.aryn.ai/?name=ArynPartitioningService_Intro>`_. 

Sycamore is a document processing engine licensed covered under the Apache v2.0 license. It's built for complex unstructured data, such as documents, presentations, transcripts, embedded tables, and internal knowledge repositories. Sycamore provides a declarative dataflow abstraction called DocSets to make manipulating unstructured documents easy and scalable. It’s similar in style to Apache Spark and Pandas, but for collections of unstructured documents. DocSets can be used not only for extracting, enriching, summarizing, and cleaning unstructured data, but also for running powerful analytics on these datasets. 

Sycamore uses LLM-powered transforms, and you can choose the model to leverage. It can handle complex documents with embedded tables, figures, graphs, and other infographics. For ETL use cases, Sycamore reliably generates vector embeddings with the model of your choice, and loads vector databases and search engines like Pinecone, OpenSearch, Weaviate, Elasticsearch, and more.  

.. image:: images/SycamoreDiagram2.png

**Key Features**

* **More powerful and accurate document segmentation for complex documents with tables, images, text, infographics, and more.**  Aryn’s vision-based segmentation models provide 11+ class labels, and are up to 6x better in mean average precision (mAP) and 4x better in mean average recall (mAR) than existing solutions. 

* **Developer-focused reliable and flexible document processing engine with Sycamore.** Similar to Apache Spark, but for handling and processing unstructured document collections at scale. Easily process document collections using sophisticated data transforms and LLMs, while maintaining overall document lineage using Sycamore’s DocSet abstraction. Create better chunks and extract higher quality metadata, leading to 30% better recall and 2x better accuracy on real-world use cases. 

* **Scalable, fault-tolerant, and reliable loading of vector DBs and search indexes.** Generate vector embeddings using your choice of model, easily build knowledge graphs, and other output formats from your unstructured data. Targets include leading engines like Elasticsearch, OpenSearch, Weaviate, Pinecone, DuckDB, and more. Aryn can seamlessly handle millions of documents. 

* **Plug-and-Play LLMs.** Use different LLMs for entity extraction, vector embedding, and post-processing steps.**


Getting Started
--------------------

**Aryn Partitioning Service**

`Sign-up here for free<https://www.aryn.ai/get-started>`_ to get an API Key.

we will need an Aryn Cloud API key, which we can get for free at aryn.ai/get-started. You will recieve the API key in your email inbox.

Next, you can:

* **Use the Aryn Playground:** Visit `the Playground<https://play.aryn.cloud/partitioning>`_ and use the UI to see how the service segments, lables, and extracts data from your documents.

* **Use the Aryn SDK:** 

1. Install the Aryn SDK using ``pip``:

``pip install aryn-sdk``

2. Then, partition your document:

.. code-block:: python

    from aryn_sdk.partition import partition_file
    with open("mydocument.pdf", "rb") as f:
    data = partition_file(f)

..

You can specify additional options (e.g. table extraction), and a list of these options is :doc:`here </aryn_cloud/aryn_partitioning_service.html#specifying-options>`_


**Sycamore**

1. Install Sycamore with ``pip``:

``pip install sycamore-ai``

2. You can next Sycamore scripts to process your docuemnts and unstructured data. `This notebook<https://github.com/aryn-ai/sycamore/blob/main/notebooks/metadata-extraction.ipynb>`_ demonstrates a variety of Sycamore data transforms and loads an OpenSearch hybrid search index.

We recommend using the Aryn Partitioning Service with Sycamore to process PDFs, and you can `sign-up here for free<https://www.aryn.ai/get-started>`_ to get an API Key. 


More Resources
--------------------
- Join the Aryn / Sycamore Slack workspace: https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg
- Watch the intro video to the Aryn Partitioning Service: https://www.aryn.ai/?name=ArynPartitioningService_Intro
- Sign up for the Aryn Partitioning Service: https://aryn.ai/get-started
- Use the Aryn Playground to experiment with the Partitioning Service: https://play.aryn.cloud/partitioning
- View the Sycamore GitHub: https://github.com/aryn-ai/sycamore

.. toctree::
   :caption: Welcome to Sycamore
   :maxdepth: 2
   :hidden:

   /welcome_to_sycamore/get_started.md
   /welcome_to_sycamore/architecture.md
   /welcome_to_sycamore/hardware.md
   /welcome_to_sycamore/encryption.md


.. toctree::
   :caption: Aryn Cloud
   :maxdepth: 2
   :hidden:

   /aryn_cloud/aryn_partitioning_service.md


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
   /APIs/aryn-sdk.rst
