# Pinecone

[Pinecone](https://www.pinecone.io/) is a vector database that supports serverless queries. Its optimization and focus on vector embedding and retrieval, and can be a good option to host large amounts of vector data.

## Configuration for Pinecone

*Please look at the [Pinecone API documentation](https://docs.pinecone.io/home) for in-depth background on the following. We specify the essential portions for creating a new Vector DB outside of Sycamore below.*

Pinecone is accessible via its cloud infrastructure hosted on AWS and GCP. To set up a new Pinecone GRPC client connection in Python, generate a new [Pinecone API key](https://www.app.pinecone.io/), install the *pinecone* python package and run the following code:

```
from pinecone.grpc import PineconeGRPC, Vector
pinecone_client = PineconeGRPC(api_key=api_key)
```

One can now write to Pinecone indexes using the `pinecone_client` client and conduct queries from there outside of Sycamore. Note this is only required if a client wishes to conduct operations external to Sycamore on the database.

## Writing to Pinecone

To write a Docset to a Pinecone index from Sycamore, use the DocSet `docset.write.pinecone(....)` function. The Pinecone writer takes in the following arguments:

- index_name: Name of the pinecone index to ingest into, is a required parameter.
- index_spec: Cloud parameters needed by pinecone to create your index. See https://docs.pinecone.io/guides/indexes/create-an-index for additional information. Defaults to None, which assumes the index already exists, and will not modify an existing index if provided
- namespace: Namespace withing the pinecone index to ingest into. See https://docs.pinecone.io/guides/indexes/use-namespaces for additional information. Defaults to "", which is the default namespace
- dimensions: Dimensionality of dense vectors in your index. Defaults to None, which assumes the index already exists, and will not modify an existing index if provided.
- distance_metric: Distance metric used for nearest-neighbor search in your index. Defaults to "cosine", but will not modify an already-existing index
- api_key: Pinecone service API Key. Defaults to None (will use the environment variable PINECONE_API_KEY).
- kwargs: Arguments to pass to the underlying execution engine

To use the writer, call write at the end of a Sycamore pipeline as done below:

```
ds.write.pinecone(index_name=index_name, namespace=namespace, dimensions=384, index_spec=spec)
```

Note that the writer forces execution of all transforms before it, so would normally come at the end of a Sycamore pipeline. More information can be found in the {doc}`API documentation </APIs/data_preparation/docsetwriter>`.
