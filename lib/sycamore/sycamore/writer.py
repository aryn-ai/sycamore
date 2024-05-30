from typing import Callable, Optional

from pyarrow.fs import FileSystem

from sycamore import Context
from sycamore.plan_nodes import Node
from sycamore.data import Document
from sycamore.writers.file_writer import default_doc_to_bytes, default_filename, FileWriter, JsonWriter


class DocSetWriter:
    """
    Contains interfaces for writing to external storage systems, most notably OpenSearch.

    Users should not instantiate this class directly, but instead access an instance using
    :meth:`sycamore.docset.DocSet.write`
    """

    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan

    def opensearch(
        self, *, os_client_args: dict, index_name: str, index_settings: Optional[dict] = None, **resource_args
    ) -> None:
        """Writes the content of the DocSet into the specified OpenSearch index.

        Args:
            os_client_args: Keyword parameters that are passed to the opensearch-py OpenSearch client constructor.
                See more information at https://opensearch.org/docs/latest/clients/python-low-level/
            index_name: The name of the OpenSearch index into which to load this DocSet.
            index_settings: Settings and mappings to pass when creating a new index. Specified as a Python dict
                corresponding to the JSON paramters taken by the OpenSearch CreateIndex API:
                https://opensearch.org/docs/latest/api-reference/index-apis/create-index/
            resource_args: Arguments to pass to the underlying execution engine

        Example:
            The following code shows how to read a pdf dataset into a ``DocSet`` and write it out to a
            local OpenSearch index called `my_index`.

            .. code-block:: python

                os_client_args = {
                    "hosts": [{"host": "localhost", "port": 9200}],
                    "http_auth": ("user", "password"),
                }

                index_settings = {
                    "body": {
                        "settings": {
                            "index.knn": True,
                        },
                        "mappings": {
                            "properties": {
                                "embedding": {
                                    "type": "knn_vector",
                                    "dimension": 384,
                                    "method": {"name": "hnsw", "engine": "faiss"},
                                },
                            },
                        },
                    },
                }

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=UnstructuredPdfPartitioner())

                pdf.write.opensearch(
                     os_client_args=os_client_args,
                     index_name="my_index",
                     index_settings=index_settings)
        """

        from sycamore.writers import OpenSearchWriter

        os = OpenSearchWriter(
            self.plan, index_name, os_client_args=os_client_args, index_settings=index_settings, **resource_args
        )
        os.execute()

    def weaviate(
        self, *, wv_client_args: dict, collection_name: str, collection_config: Optional[dict] = None, **resource_args
    ) -> None:
        """Writes the content of the DocSet into the specified Weaviate collection.

        Args:
            wv_client_args: Keyword parameters that are passed to the weaviate client constructor.
                See more information at https://weaviate.io/developers/weaviate/client-libraries/python#python-client-v4-explicit-connection
            collection_name: The name of the Weaviate collection into which to load this DocSet.
            collection_config: Keyword parameters that are passed to the weaviate client's `collections.create()`
                method.If not provided, Weaviate will Auto-Schematize the incoming records, which may lead to
                inconsistencies or failures. See more information at
                https://weaviate.io/developers/weaviate/manage-data/collections#create-a-collection-and-define-properties
            resource_args: Arguments to pass to the underlying execution engine

        Example:
            The following code shows how to read a pdf dataset into a ``DocSet`` and write it out to a
            local Weaviate collection called `DemoCollection`.

            .. code-block:: python


                collection = "DemoCollection"
                wv_client_args = {
                    "connection_params": ConnectionParams.from_params(
                        http_host="localhost",
                        http_port=8080,
                        http_secure=False,
                        grpc_host="localhost",
                        grpc_port=50051,
                        grpc_secure=False,
                    )
                }

                # Weaviate will assume empty arrays are empty arrays of text, so it
                # will throw errors when you try to make an array of non-text in a
                # field that some records have empty. => We specify them here.
                collection_config_params = {
                    "name": collection,
                    "description": "A collection to demo data-prep with sycamore",
                    "properties": [
                        Property(
                            name="properties",
                            data_type=DataType.OBJECT,
                            nested_properties=[
                                Property(
                                    name="links",
                                    data_type=DataType.OBJECT_ARRAY,
                                    nested_properties=[
                                        Property(name="text", data_type=DataType.TEXT),
                                        Property(name="url", data_type=DataType.TEXT),
                                        Property(name="start_index", data_type=DataType.NUMBER),
                                    ],
                                ),
                            ],
                        ),
                        Property(name="bbox", data_type=DataType.NUMBER_ARRAY),
                        Property(name="shingles", data_type=DataType.INT_ARRAY),
                    ],
                    "vectorizer_config": [Configure.NamedVectors.text2vec_transformers(name="embedding")],
                    "references": [ReferenceProperty(name="parent", target_collection=collection)],
                }

                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                davinci_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
                tokenizer = HuggingFaceTokenizer(model_name)

                ctx = sycamore.init()

                ds = ctx.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=UnstructuredPdfPartitioner())
                    .regex_replace(COALESCE_WHITESPACE)
                    .extract_entity(entity_extractor=OpenAIEntityExtractor(
                            "title", llm=davinci_llm, prompt_template=title_template))
                    .mark_bbox_preset(tokenizer=tokenizer)
                    .merge(merger=MarkedMerger())
                    .spread_properties(["path", "title"])
                    .split_elements(tokenizer=tokenizer, max_tokens=512)
                    .explode()
                    .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
                    .sketch(window=17)

                ds.write.weaviate(
                    wv_client_args=wv_client_args,
                    collection_name=collection,
                    collection_config=collection_config_params
                )
        """
        from sycamore.writers import WeaviateWriter

        wv = WeaviateWriter(self.plan, collection_name, wv_client_args, collection_config, **resource_args)
        wv.execute()

    def pinecone(
        self,
        *,
        index_name,
        index_spec=None,
        namespace="",
        dimensions=None,
        distance_metric="cosine",
        api_key=None,
        **resource_args,
    ):
        """Writes the content of the DocSet into a Pinecone vector index.

        Args:
            index_name: Name of the pinecone index to ingest into
            index_spec: Cloud parameters needed by pinecone to create your index. See
                    https://docs.pinecone.io/guides/indexes/create-an-index
                    Defaults to None, which assumes the index already exists, and
                    will not modify an existing index if provided
            namespace: Namespace withing the pinecone index to ingest into. See
                    https://docs.pinecone.io/guides/indexes/use-namespaces
                    Defaults to "", which is the default namespace
            dimensions: Dimensionality of dense vectors in your index.
                    Defaults to None, which assumes the index already exists, and
                    will not modify an existing index if provided.
            distance_metric: Distance metric used for nearest-neighbor search in your index.
                    Defaults to "cosine", but will not modify an already-existing index
            api_key: Pinecone service API Key. Defaults to None (will use the environment
                    variable PINECONE_API_KEY).
            resource_args: Arguments to pass to the underlying execution engine

        Example:
            The following shows how to read a pdf dataset into a ``DocSet`` and write it out
            to a pinecone index called "mytestingindex"

            .. code-block:: python

                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                tokenizer = HuggingFaceTokenizer(model_name)
                ctx = sycamore.init()
                ds = (
                    ctx.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner(extract_table_structure=True, extract_images=True))
                    .explode()
                    .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
                    .term_frequency(tokenizer=tokenizer, with_token_ids=True)
                    .sketch(window=17)
                )

                ds.write.pinecone(
                    index_name="mytestingindex",
                    index_spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
                    namespace="",
                    dimensions=384,
                    distance_metric="dotproduct",
                )

        """
        from sycamore.writers import PineconeWriter

        pc = PineconeWriter(
            self.plan,
            index_name,
            index_spec,
            namespace,
            dimensions,
            distance_metric,
            api_key,
            **resource_args,
        )
        pc.execute()

    def files(
        self,
        path: str,
        filesystem: Optional[FileSystem] = None,
        filename_fn: Callable[[Document], str] = default_filename,
        doc_to_bytes_fn: Callable[[Document], bytes] = default_doc_to_bytes,
        **resource_args,
    ) -> None:
        """Writes the content of each Document to a separate file.

        Args:
            path: The path prefix to write to. Should include the scheme if not local.
            filesystem: The pyarrow.fs FileSystem to use.
            filename_fn: A function for generating a file name. Takes a Document
                and returns a unique name that will be appended to path.
            doc_to_bytes_fn: A function from a Document to bytes for generating the data to write.
                Defaults to using text_representation if available, or binary_representation
                if not.
            resource_args: Arguments to pass to the underlying execution environment.
        """
        file_writer = FileWriter(
            self.plan,
            path,
            filesystem=filesystem,
            filename_fn=filename_fn,
            doc_to_bytes_fn=doc_to_bytes_fn,
            **resource_args,
        )

        file_writer.execute()

    def json(
        self,
        path: str,
        filesystem: Optional[FileSystem] = None,
        **resource_args,
    ) -> None:
        """
        Writes Documents in JSONL format to files, one file per
        block.  Typically, a block corresponds to a single
        pre-explode source document.

        Args:
            path: The path prefix to write to. Should include the scheme if not local.
            filesystem: The pyarrow.fs FileSystem to use.
            resource_args: Arguments to pass to the underlying execution environment.
        """

        node = JsonWriter(self.plan, path, filesystem=filesystem, **resource_args)
        node.execute()
