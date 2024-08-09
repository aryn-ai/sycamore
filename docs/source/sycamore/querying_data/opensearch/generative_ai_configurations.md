# Generative AI configurations for queries

Sycamore uses generative AI for specific operations when querying data:

* **Query Rewriting:** Currently, this feature is available in the Demo Query UI, where it will rewrite a question using conversational context and other improvements (e.g. correcting spelling mistakes). It uses OpenAI.

* **Retrieval-Augmented Generation (RAG):** Sycamore leverages the OpenSearch Search Pipeline feature set for running RAG pipelines. You can set the LLM to use in the pipeline, or override it at query time. By default, it’s configured to use OpenAI, and you specify the model. The Demo UI has an easy dropdown to select the model.

Amazon Bedrock models and Anthropic are supported as well. To configure these, you need to configure the OpenSearch components of Sycamore directly. You can [create a connector blueprint]( https://opensearch.org/docs/latest/ml-commons-plugin/remote-models/blueprints/), [configure the model](https://opensearch.org/docs/latest/ml-commons-plugin/remote-models/index/), and [connect it to your RAG pipelines](https://opensearch.org/docs/latest/search-plugins/conversational-search/#rag-pipeline).
