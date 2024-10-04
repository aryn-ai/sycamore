# Qdrant

[Qdrant](https://https://qdrant.tech/) is an Open-Source Vector Database and Vector Search Engine written in Rust for large scale data. It provides fast and scalable vector similarity search service with convenient API.

## Configuration for Qdrant

You can refer to the [Quickstart documentation](https://qdrant.tech/documentation/quickstart/) to get yourself up and running with Qdrant.

## Writing to Qdrant

To write a Docset to a Qdrant collection in Sycamore, use the DocSet `docset.write.qdrant(....)` function. The Qdrant writer accepts the following arguments:

- `client_params`: Parameters that are passed to the Qdrant client constructor. See more information in the [Client API Reference](https://python-client.qdrant.tech/qdrant_client.qdrant_client).
- `collection_params`: Parameters that are passed into the `qdrant_client.QdrantClient.create_collection` method. See more information in the [Client API Reference](https://python-client.qdrant.tech/_modules/qdrant_client/qdrant_client#QdrantClient.create_collection).
- `vector_name`: The name of the vector in the Qdrant collection. Defaults to `None`.
- `execute`: Execute the pipeline and write to Qdrant on adding this operator. If `False`, will return a `DocSet` with this write in the plan. Defaults to `True`.
- `kwargs`: Keyword arguments to pass to the underlying execution engine.

```python
ds.write.qdrant(
    {
        "url": "http://localhost:6333",
        "timeout": 50,
    },
    {
        "collection_name": "{collection_name}",
        "vectors_config": {
            "size": 384,
            "distance": "Cosine",
        },
    },
)

```

Note that the writer forces execution of all transforms before it, so would normally come at the end of a Sycamore pipeline. More information can be found in the {doc}`API documentation <../APIs/docsetwriter>`.

## Reading from Qdrant

To read a Docset from a Qdrant collection in Sycamore, use the DocSet `docset.read.qdrant(....)` function. The Qdrant reader accepts the following arguments:

- `client_params`: Parameters that are passed to the Qdrant client constructor. See more information in the[Client API Reference](https://python-client.qdrant.tech/qdrant_client.qdrant_client).
- `query_params`: Parameters that are passed into the `qdrant_client.QdrantClient.query_points` method. See more information in the [Client API Reference](https://python-client.qdrant.tech/_modules/qdrant_client/qdrant_client#QdrantClient.query_points).
- `kwargs`: Keyword arguments to pass to the underlying execution engine.

```python
docs = ctx.read.qdrant(
    {
        "url": "https://xyz-example.eu-central.aws.cloud.qdrant.io:6333",
        "api_key": "<paste-your-api-key-here>",
    },
    {"collection_name": "{collection_name}", "limit": 100, "using": "{optional_vector_name}"},
).take_all()

```

More information can be found in the {doc}`API documentation <../APIs/docsetreader>`.
