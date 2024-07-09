## Embed
The Embed Transform is responsible for generating embeddings for your Documents or Elements. These embeddings are stored in a special ``embedding`` property on each document.

This Embed transform takes a single argument -- the `embedder`, which encapsulates a specific embedding model and it's parameters. The currently supported models are listed below. More information can be found in the {doc}`API documentation </APIs/transforms/embed>`.

### SentenceTransformers

The `SentenceTransformerEmbedder` embeds the text representation of each document using any of the models from the popular [SentenceTransformers framework](https://www.sbert.net/). The embeddings are computed locally, and Sycamore will automatically batch records and leverage GPUs where appropriate.

The following exmaple code embeds a ``DocSet`` with the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model:

```python
embedder = SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
embedded_doc_set = docset.embed(embedder)
```

### OpenAI Embeddings

The `OpenAIEmbedder` embeds the text representation of each document using the `text-embedding-ada-002` model. The `model_batch_size` parameter controls how many documents are sent to the OpenAI endpoint as a single call. For example, the following snippet will send records in batches of 1000 to OpenAI, using the default embedding model (`text-embedding-ada-002`).

```python
embedded_doc_set = docset.embed(OpenAIEmbedder(model_batch_size=1000))
```

By default the transform will look for an OpenAI API key in the `OPENAI_API_KEY` environment variable. It can also be optionally passed in via the `api_key` parameter.


### Amazon Bedrock Embeddings

The `BedrockEmbedder` calls the Amazon Bedrock service to compute embeddings. Currently the only supported embedding model in Amazon Bedrock is `amazon.titan-embed-text-v1`. Sycamore makes its API calls to Bedrock using the `boto3` library. Since Sycamore will compute embeddings in parallel, rather than passing in boto3 client directly, you pass in the arguments necessary to construct a [boto3 Session object](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html). This is most often used to configure AWS credentials. For example, to use a specific access key id and secret access key, you could call embed as follows:

```python
embedded_doc_set = docset.embed(BedrockEmbedder(boto_session_kwargs={
    "aws_access_key_id": "<access_key>",
    "aws_secret_access_key":
    "<secret access key>"}))
```

Sycamore will then construct a boto3 Session object on each executor using the specified credentials. If you do not specify credentials the standard credential resolution mechanisms are used. More information on AWS credentials can be found [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

The bedrock APIs do not support batching, so an API call will be made for each document in the `DocSet`.
