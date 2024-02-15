Data Preparation
============

In Sycamore, a transform is a method that operates on a ``DocSet`` and returns a new ``DocSet``. Sycamore provides a number of these transforms directly in the ``DocSet`` 
class to prepare and enhance your unstructured data. In order to support a variety of data types and machine learning models, many of these transforms are customizable with different implementations.


.. toctree::
   :maxdepth: 1

   ./transforms/embed.md
   ./transforms/explode.md
   ./transforms/extract_entity.md
   ./transforms/extract_schema.md
   ./transforms/flatmap.md
   ./transforms/map.md
   ./transforms/map_batch.md
   ./transforms/partition.md
   ./transforms/summarize.md
