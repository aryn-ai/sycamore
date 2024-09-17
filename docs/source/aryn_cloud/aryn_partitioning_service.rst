Aryn Partitioning Service
=========================

You can use the Aryn Partitioning Service to segment PDFs and Microsoft Office (.doc, .docx, .ppt, .pptx) documents and to automatically extract table data.  The service labels titles, tables, table rows and columns, images, and regular text. Bounding boxes are returned as JSON with their associated text for easy use. It leverages `Aryn's purpose-built AI model for document segmentation and labelling <https://huggingface.co/Aryn/deformable-detr>`_ that was trained using DocLayNet – an open source, human-annotated document layout segmentation dataset containing tens of thousands of pages from a broad variety of document sources.


If you'd like to experiment with the service, you can use the UI in the `Aryn Playground <https://play.aryn.cloud/partitioning>`_ to visualize how your documents will be partitioned. To get started with the Aryn Partitoning Service, please check out the following pages:

.. toctree::
   :maxdepth: 1

   ./get_started.md
   ./accessing_the_partitioning_service.rst
   ./specifying_options.rst
   ./aps_output.md
   ./using_the_console.md
