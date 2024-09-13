# ETL tutorial with Sycamore and Pinecone

[This tutorial](https://colab.research.google.com/drive/1oWi50uqJafBDmLWNO4QFEbiotnU7o75B) is meant to show how to create an ETL pipeline with Sycamore to load a Pinecone vector database. It walks through an intermediate-level ETL flow: partitioning, extraction, cleaning, chunking, embedding, and loading. You will need an [Aryn Partitioning Service API key](https://www.aryn.ai/get-started), [OpenAI API key](https://platform.openai.com/signup) (for LLM-powered data enrichment and creating vector embeddings), and a [Pinecone API key](https://app.pinecone.io/?sessionType=signup) (for creating and using a vector index). At the time of writing, there are free trial or free tier options for all of these services.

Run this tutorial in a [Colab notebook](https://colab.research.google.com/drive/1oWi50uqJafBDmLWNO4QFEbiotnU7o75B) or [locally with Jupyter](https://github.com/aryn-ai/sycamore/blob/main/notebooks/sycamore-tutorial-intermediate-etl.ipynb).

Once you have your data loaded in Pinecone, you can use Pinecone's query features for semantic search or a framework like Langchain for RAG. The [Pinecone Writer example notebook](https://github.com/aryn-ai/sycamore/blob/main/notebooks/pinecone-writer.ipynb) has sample Langchain code at the end of the notebook.
