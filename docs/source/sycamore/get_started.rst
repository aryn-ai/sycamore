Installing and Getting Started With Sycamore
=========================

Install Library
---------------

We recommend installing the Sycamore library using ``pip``:

.. code-block:: python

   pip install sycamore-ai

..

Connectors for vector databases can be installed via extras. For example,

.. code-block:: python

    pip install sycamore-ai[opensearch]

..

will install Sycamore with OpenSearch support. You can find a list of supported connectors :doc:`here </sycamore/connectors>`.

By default, Sycamore works with Aryn DocParse to process PDFs. To run inference locally, install the `local-inference` extra as follows:

.. code-block:: python

    pip install sycamore-ai[local-inference]

..

Next, you can set the proper API keys for related services, like Aryn DocParse for processing PDFs (`sign-up here <https://www.aryn.ai/get-started>`_ for free) or OpenAI to use GPT with Sycamore's LLM-based transforms.

Now, that you have installed Sycamore, you see it in action using the example Jupyter notebooks. Many of these examples load a vector database in the last step of the processing pipeline, but you can edit the notebook to write the data to a different target database or out to a file. `Visit the Sycamore GitHub <https://github.com/aryn-ai/sycamore/tree/main/notebooks>`_ for the sample notebooks.

Here are a few good notebooks to start with:

* An `intermediate ETL tutorial notebook <https://github.com/aryn-ai/sycamore/blob/main/notebooks/sycamore-tutorial-intermediate-etl.ipynb>`_ walking through an ETL flow with chunking (using DocParse), LLM-based data enrichment, data cleaning, and loading a Pinecone hybrid search index
* A `notebook <https://github.com/aryn-ai/sycamore/blob/main/notebooks/tutorial.ipynb>`_ showing a simple processing job using DocParse to chunk PDFs, two LLM-based entity extraction transforms, and loading an OpenSearch hybrid index (vector + keyword)
* A `notebook <https://github.com/aryn-ai/sycamore/blob/main/notebooks/VisualizePartitioner.ipynb>`_ that visually shows the bounding boxes created by Aryn DocParse.
* A `more advanced Sycamore pipeline <https://github.com/aryn-ai/sycamore/blob/main/notebooks/metadata-extraction.ipynb>`_ that chunks PDFs using DocParse, does schema extraction and population using LLM transforms, data cleaning using Python, and loads an OpenSearch hybrid index (vector + keyword)
* A `notebook <https://github.com/aryn-ai/sycamore/blob/main/notebooks/pinecone-writer.ipynb>`_ showing how to load a Pinecone vector database. There are other example notebooks showing sample code for loading other targets `here <https://github.com/aryn-ai/sycamore/tree/main/notebooks>`_.



.. toctree::
   :maxdepth: 1

   ./get_started/concepts.md
   ./get_started/hardware.md
   ./get_started/ai_configuration.md
   
