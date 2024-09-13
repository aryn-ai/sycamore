Specifying Options
=============

There are several options you can specify when calling the partitioning service. For example, we can extract the table structure from our document with the following curl command.

.. code:: bash

    export ARYN_API_KEY="PUT API KEY HERE"
    curl -s -N -D headers "https://api.aryn.cloud/v1/document/partition" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.pdf" -F 'options={"extract_table_structure": true}' | tee document.json

All of the available options are listed below:

* ``threshold``: This represents the threshold for accepting the model's predicted bounding boxes. It defaults to "auto", where the model make its best prediction. The user can specify a numerical threshold between 0 and 1. A lower value will include more objects, but may have overlaps, a higher value will reduce the number of overlaps, but may miss legitimate objects. We recommend using the default "auto" threshold. If you do set the threshold manually, we recommend starting with a value of 0.32.
* ``use_ocr``: It defaults to ``false``, where the partitioner attempts to directly extract the text from the underlying PDF using PDFMiner.  If ``true``, the partitioner detects and extracts text using Tesseract, an open source OCR library.
* ``extract_table_structure``: If ``true``, the partitioner runs a table extraction model separate from the segmentation model in order to extract cells from regions of the document identified as tables.
* ``extract_images``: If ``true``, the partitioner crops each region identified as an image and attaches it to the associated ``ImageElement``. This can later be fed into the ``SummarizeImages`` transform when used within Sycamore.
* ``selected_pages``: You can specify a page (like ``[11]`` ), a page range (like ``[[25,30]]`` ), or a combination of both (like ``[11, [25,30]]`` ) of your PDF to process. The first page of the PDF is ``1``, not ``0``.
* ``pages_per_call``: This is only available when using the Partition function in Sycamore. This option divides the processing of your document into batches of pages, and you specify the size of each batch (number of pages). This is useful when running OCR on large documents. 

You can use multiple options at the same time like in the example below:

.. code:: bash

    export ARYN_API_KEY="PUT API KEY HERE"
    curl -s -N -D headers "https://api.aryn.cloud/v1/document/partition" -H "Authorization: Bearer $ARYN_API_KEY" -F "pdf=@document.pdf" -F 'options={"extract_table_structure": true, "threshold": 0.2}' | tee document.json
