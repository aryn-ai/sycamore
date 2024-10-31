Welcome to Sycamore!
================
Aryn is an LLM-powered data preparation, processing, and analytics system for complex, unstructured documents like PDFs, HTML, presentations, and more. With Aryn, you can prepare data for GenAI and RAG applications, power high-quality document processing workflows, and run analytics on large document collections with natural language. It includes three components: Aryn DocParse, Aryn DocPrep, and Sycamore. For Aryn DocParse and Aryn DocPrep, please visit the `Aryn DocParse documentation <https://docs.aryn.ai>`_.

Sycamore is a document processing engine covered under the Apache v2.0 license. It's built for complex unstructured data, such as documents, presentations, transcripts, embedded tables, and internal knowledge repositories. Sycamore provides a declarative dataflow abstraction called a DocSet to make manipulating unstructured documents easy and scalable. It’s similar in style to Apache Spark and Pandas, but for collections of unstructured documents. DocSets can be used not only for extracting, enriching, summarizing, and cleaning unstructured data, but also for running powerful analytics on these datasets.

Sycamore uses LLM-powered transforms, and you can choose the model to leverage. It can handle complex documents with embedded tables, figures, graphs, and other infographics. For ETL use cases, Sycamore reliably generates vector embeddings with the model of your choice, and loads vector databases and search engines like Pinecone, OpenSearch, Weaviate, Elasticsearch, Qdrant and more.

.. image:: images/ArynArchitecture_APS+Sycamorev2.png

Getting Started
--------------------

Sycamore
^^^^^^^^

1. Install Sycamore with ``pip``:

.. code-block:: python

    pip install sycamore-ai

..

Support for vector databases can be installed using extras. For example,

.. code-block:: python

    pip install sycamore-ai[opensearch]

..

will install Sycamore with OpenSearch support. You can find a list of supported connectors :doc:`here </sycamore/connectors>`.

By default, Sycamore works with the Aryn Partitioning Service to process PDFs. you can `sign-up here for free <https://www.aryn.ai/get-started>`_ to get an API Key. To install support for local partitioning and embedding models, you can install Sycamore with the ``local-inference`` extra:

.. code-block:: python

    pip install sycamore-ai[local-inference]

..


2. You can now create and run Sycamore scripts to process your documents and unstructured data. `This notebook <https://github.com/aryn-ai/sycamore/blob/main/notebooks/metadata-extraction.ipynb>`_ demonstrates a variety of Sycamore data transforms and loads an OpenSearch hybrid search index.


More Resources
--------------------
- Visit the Aryn DocParse and DocPrep Documentation: https://docs.aryn.ai/introduction
- Join the Aryn / Sycamore Slack workspace: https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg
- Watch the intro video to the Aryn Partitioning Service: https://www.aryn.ai/?name=ArynPartitioningService_Intro
- Sign up for the Aryn Partitioning Service: https://aryn.ai/get-started
- Use the Aryn Playground to experiment with the Partitioning Service: https://play.aryn.cloud/partitioning
- View the Sycamore GitHub: https://github.com/aryn-ai/sycamore

.. toctree::
   :maxdepth: 2
   :hidden:

   /sycamore/get_started.rst
   /sycamore/using_jupyter.md
   /sycamore/transforms.rst
   /sycamore/connectors.rst
   /sycamore/query.rst
   /sycamore/tutorials.rst
   /sycamore/APIs.rst
   /sycamore/SDK_APIs.rst
