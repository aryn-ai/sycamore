Get Started With Sycamore
=========================

Install Library
---------------

We recommend installing the Sycmaore library using ``pip``:

.. code-block:: python

   pip install sycamore-ai

..

If you are using Sycamore in a notebook environment, you can optionally install [Poppler](https://poppler.freedesktop.org/) to visualize PDFs and the bounding boxes and labels Aryn uses to partition it.

Next, you can set the proper API keys for related services, like the Aryn Partitioning Service (APS) for processing PDFs (`sign-up here <https://www.aryn.ai/get-started>`_ for free) or OpenAI to use GPT with Sycamore's LLM-based transforms.

Now, that you have installed Sycamore, you see it in action using the example Jupyter notebooks. Many of these examples load a vector database in the last step of the processing pipeline, but you can edit the notebook to write the data to a different target database or out to a file. `Visit the Sycamore GitHub <https://github.com/aryn-ai/sycamore/tree/main/notebooks>`_ for the sample notebooks.

Here are a few good notebooks to start with:

* A `notebook <https://github.com/aryn-ai/sycamore/blob/main/notebooks/tutorial.ipynb>`_ showing a simple processing job using APS to chunk PDFs, two LLM-based entity extraction transforms, and loading an OpenSearch hybrid index (vector + keyword)
* A `notebook <https://github.com/aryn-ai/sycamore/blob/main/notebooks/VisualizePartitioner.ipynb>`_ that visually shows the bounding boxes created by the Aryn Partioning Service
* A `more advanced Sycamore pipeline <https://github.com/aryn-ai/sycamore/blob/main/notebooks/metadata-extraction.ipynb>`_ that chunks PDFs using APS, does schema extraction and population using LLM transforms, data cleaning using Python, and loads an OpenSearch hybrid index (vector + keyword)
* A `notebook <https://github.com/aryn-ai/sycamore/blob/main/notebooks/pinecone-writer.ipynb>`_ showing how to load a Pinecone vector database. There are other example notebooks showing sample code for loading other targets `here <https://github.com/aryn-ai/sycamore/tree/main/notebooks>`_.



.. toctree::
   :maxdepth: 1

   ./get_started/concepts.md
   ./get_started/hardware.md
   ./get_started/ai_configuration.md
   
