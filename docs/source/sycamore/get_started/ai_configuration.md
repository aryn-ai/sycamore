# AI Configuration

## Aryn Partitioner

The Aryn Partitioner is the recommended way to process PDF documents in Sycamore, and it uses a state-of-the-art, open source deep learning AI model trained on 80k+ enterprise documents. By default, it's configured to use [Aryn DocParse](https://docs.aryn.ai/quickstart), and you will need to set your API key. You can sign-up for free [here](https://www.aryn.ai/get-started) to get an API key for the service.


## LLM-based Transforms
Sycamore brings generative AI to a variety of stages in an ETL pipeline. You can choose different LLMs for entity extraction, schema extraction, and more. Currently, Sycamore supports OpenAI and Amazon Bedrock, and you will need to set your credentials for these services for Sycamore to use.

Information on supported generative AI models for each operation are in the specific documentation for it:

* [Entity extraction](./transforms/extract_entity.md)
* [Schema extraction](./transforms/extract_schema.md)
* [Summarize](./transforms/summarize.md)

## Creating Vector Embeddings
A final step before loading a vector database is creating vector embeddings for your data. Currently, Sycamore supports OpenAI and Amazon Bedrock, and you will need to set your credentials for these services for Sycamore to use.

For more information on creating embeddings, visit [here](./transforms/embed.md).