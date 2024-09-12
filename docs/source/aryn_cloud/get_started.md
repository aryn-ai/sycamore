# An Introduction to the Aryn Partitioning Service 
You can use the Aryn Partitioning Service to easily chunk and extract data from complex PDFs. The Partitioning Service can extract paragraphs, tables and images. It returns detailed information about the components it has just identified in a JSON object.  The following two sections will walk you through two examples where we segment  PDF documents and extract a table and an image from those documents using the python aryn-sdk.

- [Table Extraction from PDF](get_started_Table_extraction.md)
- [Image Extraction from PDF](get_started_Image_extraction.md)



### More examples


#### Using the Partitioning Service with Sycamore

You can  checkout a notebook [here](https://github.com/aryn-ai/sycamore/blob/main/notebooks/pinecone-writer.ipynb) to learn how to use the partitioning service with Aryn's Sycamore analytics engine. This notebook walks through an example where you can use Sycamore to transform your data and load it into a vector database.

#### Using the Partitioning Service with Langchain

You can  checkout a notebook [here](https://github.com/aryn-ai/sycamore/blob/main/notebooks/ArynPartitionerWithLangchain.ipynb) to learn how to use the partitioning service with Langchain.
