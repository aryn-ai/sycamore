.. image:: images/sycamore_logo.svg
   :alt: Sycamore
====================================


Welcome to Sycamore!
--------------------

Sycamore is an open source semantic ETL system for building sophisticated search applications. With Sycamore, you can easily understand the structure of complex documents, use LLMs to extract semantic information, and create vector embeddings for semantic search.

Sycamore has some key properties make it particularly well-suited for these tasks.

- **Set-based abstraction for unstructured documents.** Sycamore introduces an abstraction called the *DocSet* to represent a collection of unstructured documents. You can think of docsets like Dataframes in Apache Spark, except that they are designed specifically for complex unstructured documents. With docsets, you can easily apply operations to a large collection of documents or perform transformations directly on the entire collection.

- **Scalable dataflow execution platform.** Sycamore executes on the `Ray <https://ray.io>`_ distributed compute framework. Ray is purpose built for running machine learning applications at scale and provides fine-grained scheduling on both CPUs and GPUs. Sycamore leverages Ray to scale data preparation pipelines automatically.

- **Easy integration with LLMs.** Sycamore makes it easy to incorporate LLMs into your dataflow pipeline where appropriate for operations like rich entity extraction.


Getting Started
--------------------

     pip install sycamore-ai

For certain PDF processing operations, you also need to install `poppler`, which you can do with the OS-native package manager of your choice. For example, the command for Homebrew on Mac OS is ``brew install poppler``



More Resources
--------------------
- Join the Sycamore Slack workspace: `Link <https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg>`_
- View the `Aryn docs <https://docs.aryn.ai>`_ to learn more about how to built end-to-end conversational search with Sycamore and OpenSearch.


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

