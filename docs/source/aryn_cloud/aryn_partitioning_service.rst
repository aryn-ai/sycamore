Aryn Partitioning Service
=========================

You can use the Aryn Partitioning Service to segment documents into labeled bounding boxes identifying titles, tables, table rows and columns, images, and regular text. The service can segment PDFs, Microsoft Word (.docx and .doc) documents, and Microsoft PowerPoint (.pptx and .ppt) file formats. It is often used to :doc:`extract structured table data </aryn_cloud/get_started_Table_Extraction>`. Bounding boxes are returned as JSON with their associated text for easy use. It leverages `Aryn's purpose-built AI model for document segmentation and labelling <https://huggingface.co/Aryn/deformable-detr>`_ that was trained using DocLayNet – an open source, human-annotated document layout segmentation dataset containing tens of thousands of pages from a broad variety of document sources.


If you'd like to experiment with the service, you can use the UI in the `Aryn Playground <https://play.aryn.cloud/partitioning>`_ to visualize how your documents will be partitioned. To get started with the Aryn Partitoning Service, please check out the following pages:

.. toctree::
   :maxdepth: 1

   ./get_started.md
   ./accessing_the_partitioning_service.rst
   ./specifying_options.rst
   ./get_started_Image_Extraction.md
   ./get_started_Table_Extraction.md
   ./aps_output.md
   ./using_the_console.md
