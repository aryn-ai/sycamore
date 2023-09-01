# Sycamore

**Open Source Semantic ETL Framework For Natural Language Search Apps**

Sycamore is an open source semantic ETL framework purpose-built for unlocking the semantic meaning of unstructured data
and preparing it for search applications. It provides a high-level API in Python to construct pipelines for data
extraction, enrichment, vector embedding, and loading into OpenSearch, a leading Apache v2.0 licensed search platform
with vector database and indexing capabilities.

- Document type support: Current support for PDF and HTML files with text, images, and tables. Word documents, Excel
  sheets, and PowerPoint decks coming soon.

- AI-enabled entity extraction: Use the large language model (LLM) of your choice, including LLAMA 2, OpenAI GPT-4, or
  Falcon, for extracting information from documents to improve semantic understanding and provide more relevant search
  results.

- Pre-built transforms: Easily prepare documents for search applications with functions built for chunking,
  manipulating, and adding search factes for unstructured data.

- Create vector embeddings: Seamlessly use a variety of LLMs, including miniLM, OpenAI GPT-4, or LLAMA 2, to create
  vector embeddings and embed them with your documents.

- Load OpenSearch: Load processed data into OpenSearch for semantic/vector and keyword (hybrid) search.

- Scale up: Integration with Apache Ray, an open source compute framework, for scaling up your processing workloads.

- Data sources: Ingest data from Amazon S3. Connectors to more data sources are coming soon, and please create an issue
  here for requesting a new source.

For more information visit [aryn.ai](https://www.aryn.ai).
