Accessing The Aryn Partitioning Service
=============

There are three ways to use the Aryn Partitioning Service: through the ``aryn-sdk`` client, through the ``ArynPartitioner`` in Sycamore, and directly from the HTTP service.

To follow along below, we will need an Aryn Cloud API key, which we can get for free at `aryn.ai/get-started <https://www.aryn.ai/get-started>`_. You will recieve the API key in your email inbox.

++++++++++++++++++
Using ``aryn-sdk``
++++++++++++++++++

The ``aryn-sdk`` client is a thin python library that calls the Aryn Partitioning Service and provides a few utility methods around it. It is the easiest way to add the Aryn Partitioning Service to your applications or custom data processing pipelines. You can view an example in `this notebook <https://github.com/aryn-ai/sycamore/blob/main/notebooks/ArynPartitionerPython.ipynb>`_.

Install the ``aryn-sdk`` client with ``pip install aryn-sdk``.
Partition a document like so:

.. code:: python

    from aryn_sdk.partition import partition_file
    with open("mydocument.pdf", "rb") as f:
        data = partition_file(f)

``partition_file`` takes the same options as curl, except as keyword arguments. You can find a list of these options `here <https://sycamore.readthedocs.io/en/stable/aryn_cloud/aryn_partitioning_service.html#specifying-options>`_.

Key management
++++++++++++++

By default, ``aryn-sdk`` looks for Aryn API keys first in the environment variable ``ARYN_API_KEY``, and then in ``~/.aryn/config.yaml``. You can override this behavior by specifying a key directly or a different path to the Aryn config file:

.. code:: python

    from aryn_sdk.partition import partition_file
    from aryn_sdk.config import ArynConfig
    with open("mydocument.pdf", "rb") as f:
        data = partition_file(f, aryn_api_key="YOUR-API-KEY")
    with open("mydocument.pdf", "rb") as f:
        data = partition_file(f, aryn_config=ArynConfig(aryn_config_path="~/dotfiles/.aryn/config.yaml"))

Helper Functions
++++++++++++++++

``aryn_sdk`` provides some helper functions to make working with and visualizing the output of ``partition_file`` easier.

.. code:: python

    from aryn_sdk.partition import partition_file, table_elem_to_dataframe, draw_with_boxes
    with open("mydocument.pdf", "rb") as f:
        data = partition_file(f, extract_table_structure=True, use_ocr=True, extract_images=True, threshold=0.35)

    # Produce a pandas DataFrame representing one of the extracted tables
    table_elements = [elt for elt in data['elements'] if elt['type'] == 'table']
    dataframe = table_elem_to_dataframe(table_elements[0])

    # Draw the detected bounding boxes on the pages. requires poppler
    images = draw_with_boxes("mydocument.pdf", data)

Different Formats
+++++++++++++++++

It is easy to use multiple file types with aryn-sdk:

.. code :: python

    from aryn_sdk.partition import partition_file
    with open("mydocument.pdf", "rb") as f:
        data = partition_file(f)
    with open("mydocument.docx", "rb") as f:
        data = partition_file(f)
    with open("mypresentation.doc", "rb") as f:
        data = partition_file(f)
    with open("mypresentation.pptx", "rb") as f:
        data = partition_file(f)
    with open("mypresentation.ppt", "rb") as f:
        data = partition_file(f)

++++++++++++++++++++++++++++++++++++
Using Sycamore's Partition transform
++++++++++++++++++++++++++++++++++++

The Aryn Partitining Service is the default option when specifying the Aryn Partitioner in a Sycamore script. Say you have a set of pdfs located at the path stored in ``work_dir``. We partition these documents with the code snippet below:

.. code:: python

    aryn_api_key = "PUT API KEY HERE"

    ctx = sycamore.init()
    pdf_docset = context.read.binary(work_dir, binary_format="pdf")
    partitioned_docset = pdf_docset.partition(ArynPartitioner(aryn_api_key=aryn_api_key))

Alternatively, we can store our Aryn API key in ``~/.aryn/config.yaml`` like so:

.. code:: yaml

    aryn_token: "PUT API KEY HERE"

Which makes our Sycamore script the following:

.. code:: python

    ctx = sycamore.init()
    pdf_docset = context.read.binary(work_dir, binary_format="pdf")
    partitioned_docset = pdf_docset.partition(ArynPartitioner())


If you are processing a large PDF with OCR, you might benefit from using the ``pages_per_call`` option. This is only available when using the Partition function in Sycamore. This option divides the processing of your document into batches of pages, and you specify the size of each batch.

Using ``curl``
++++++++++++++

We recommend using the Aryn SDK, but you can also use ``curl`` to access the Aryn Partitioning Service directly.

``curl`` an example document to use with the partitioning service if you do not have one already.

.. code:: bash

    curl http://arxiv.org/pdf/1706.03762 -o document.pdf

Change ``PUT API KEY HERE`` below to your Aryn API key. If you have a different document, change ``@document.pdf`` to ``@/path/to/your/document.pdf`` below.

.. code:: bash

    export ARYN_API_KEY="PUT API KEY HERE"
    curl -s -N -D headers "https://api.aryn.cloud/v1/document/partition" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.pdf" | tee document.json

Your results have been saved to ``document.json``.

.. code:: bash

    cat document.json

Different Formats
+++++++++++++++++

.. code:: bash

    export ARYN_API_KEY="PUT API KEY HERE"
    curl -s -N -D headers "https://api.aryn.cloud/v1/document/partition" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.pdf" | tee document.json
    curl -s -N -D headers "https://api.aryn.cloud/v1/document/partition" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.docx" | tee document.json
    curl -s -N -D headers "https://api.aryn.cloud/v1/document/partition" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.doc" | tee document.json
    curl -s -N -D headers "https://api.aryn.cloud/v1/document/partition" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.pptx" | tee document.json
    curl -s -N -D headers "https://api.aryn.cloud/v1/document/partition" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.ppt" | tee document.json