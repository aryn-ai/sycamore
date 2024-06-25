from typing import Any, Callable, Optional, TYPE_CHECKING

from pyarrow.fs import FileSystem

from sycamore import Context
from sycamore.plan_nodes import Node
from sycamore.data import Document
from sycamore.writers.common import HostAndPort
from sycamore.writers.file_writer import default_doc_to_bytes, default_filename, FileWriter, JsonWriter
import os
import duckdb
import glob

if TYPE_CHECKING:
    # Shenanigans to avoid circular import
    from sycamore.docset import DocSet


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
        self,
        *,
        os_client_args: dict,
        index_name: str,
        index_settings: Optional[dict] = None,
        execute: bool = True,
        **kwargs,
    ) -> Optional["DocSet"]:
        """Writes the content of the DocSet into the specified OpenSearch index.

        Args:
            os_client_args: Keyword parameters that are passed to the opensearch-py OpenSearch client constructor.
                See more information at https://opensearch.org/docs/latest/clients/python-low-level/
            index_name: The name of the OpenSearch index into which to load this DocSet.
            index_settings: Settings and mappings to pass when creating a new index. Specified as a Python dict
                corresponding to the JSON paramters taken by the OpenSearch CreateIndex API:
                https://opensearch.org/docs/latest/api-reference/index-apis/create-index/
            execute: Execute the pipeline and write to opensearch on adding this operator. If false,
                will return a new docset with the write in the plan
            kwargs: Arguments to pass to the underlying execution engine

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

        from sycamore.connectors.opensearch import OpenSearchWriter, OpenSearchClientParams, OpenSearchTargetParams
        from typing import Any
        import copy

        # We mutate os_client_args, so mutate a copy
        os_client_args = copy.deepcopy(os_client_args)

        # Type narrowing for hosts joy
        def _convert_to_host_port_list(hostlist: Any) -> list[HostAndPort]:
            if not isinstance(hostlist, list):
                raise ValueError('OpenSearch client args "hosts" param must be a list of hosts')
            for h in hostlist:
                if (
                    not isinstance(h, dict)
                    or "host" not in h
                    or not isinstance(h["host"], str)
                    or "port" not in h
                    or not isinstance(h["port"], int)
                ):
                    raise ValueError(
                        'OpenSearch client args "hosts" objects must consist of dicts of '
                        "the form {'host': '<address>', 'port': <port num>}\n"
                        f"Found: {h}"
                    )
            return [HostAndPort(host=h["host"], port=h["port"]) for h in hostlist]

        hosts = os_client_args.get("hosts", None)
        if hosts is not None:
            os_client_args["hosts"] = _convert_to_host_port_list(hosts)
        client_params = OpenSearchClientParams(**os_client_args)

        target_params: OpenSearchTargetParams
        if index_settings is not None:
            idx_settings = index_settings.get("body", {}).get("settings", {})
            idx_mappings = index_settings.get("body", {}).get("mappings", {})
            target_params = OpenSearchTargetParams(index_name, idx_settings, idx_mappings)
        else:
            target_params = OpenSearchTargetParams(index_name, {}, {})

        os = OpenSearchWriter(
            self.plan, client_params=client_params, target_params=target_params, name="OsrchWrite", **kwargs
        )

        # We will probably want to break this at some point so that write
        # doesn't execute automatically, and instead you need to say something
        # like docset.write.opensearch().execute(), allowing sensible writes
        # to multiple locations and post-write operations.
        if execute:
            # If execute, force execution
            os.execute().materialize()
            return None
        else:
            from sycamore.docset import DocSet

            return DocSet(self.context, os)

    def weaviate(
        self,
        *,
        wv_client_args: dict,
        collection_name: str,
        collection_config: Optional[dict[str, Any]] = None,
        execute: bool = True,
        **kwargs,
    ) -> Optional["DocSet"]:
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
        from sycamore.writers.weaviate_writer import (
            WeaviateDocumentWriter,
            WeaviateCrossReferenceWriter,
            WeaviateClientParams,
            WeaviateTargetParams,
            CollectionConfigCreate,
        )

        if collection_config is None:
            collection_config = dict()
        client_params = WeaviateClientParams(**wv_client_args)
        collection_config_object: CollectionConfigCreate
        if "name" in collection_config:
            assert collection_config["name"] == collection_name
            collection_config_object = CollectionConfigCreate(**collection_config)
        else:
            collection_config_object = CollectionConfigCreate(name=collection_name, **collection_config)
        target_params = WeaviateTargetParams(name=collection_name, collection_config=collection_config_object)

        wv_docs = WeaviateDocumentWriter(
            self.plan, client_params, target_params, name="weaviate_write_documents", **kwargs
        )
        wv_refs = WeaviateCrossReferenceWriter(
            wv_docs, client_params, target_params, name="weaviate_write_references", **kwargs
        )

        if execute:
            # If execute, force execution
            wv_refs.execute().materialize()
            return None
        else:
            from sycamore.docset import DocSet

            return DocSet(self.context, wv_refs)

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

    def duckdb(
        self,
        db_url: Optional[str] = None,
        table_name: Optional[str] = None,
        execute: bool = True,
        csv_directory_location: Optional[str] = None,
        **kwargs,
    ):
        """
        Writes the content of the DocSet into a DuckDB database.

        Args:
            db_url: The URL of the DuckDB database. If not provided, the database will be in-memory.
            table_name: The table name to write the data to when possible
            execute: Flag that determines whether to execute immediately
            csv_directory_location: The location to write the csv files to. If not provided, defaults to "./tmp/duckdb"
            and is removed after execution.

        Example:
            The following shows how to read a pdf dataset into a ``DocSet`` and write it out
            to a DuckDB database and read from it.

            .. code-block:: python
                table_name = "duckdb_table"
                db_url = ":default:"
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                paths = str(TEST_DIR / "resources/data/pdfs/")

                OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
                tokenizer = HuggingFaceTokenizer(model_name)

                ctx = sycamore.init(ray_args={"runtime_env": {"worker_process_setup_hook": ray_logging_setup}})

                ds = (
                    ctx.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=UnstructuredPdfPartitioner())
                    .regex_replace(COALESCE_WHITESPACE)
                    .mark_bbox_preset(tokenizer=tokenizer)
                    .merge(merger=MarkedMerger())
                    .spread_properties(["path"])
                    .split_elements(tokenizer=tokenizer, max_tokens=512)
                    .explode()
                    .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
                )
                ds.write.duckdb(table_name=table_name, db_url=db_url)
                conn = duckdb.connect(database=db_url)
                duckdb_read = conn.execute(f"SELECT * FROM {table_name}")
        """
        from sycamore.writers.duckdb_csv_writer import DuckDBCSVWriter, DuckDBClientParams, DuckDBTargetParams

        csv_location = csv_directory_location if csv_directory_location is not None else "tmp"
        client_params = DuckDBClientParams()
        target_params = DuckDBTargetParams(parquet_location=csv_location)
        ddb_csv = DuckDBCSVWriter(
            self.plan,
            client_params=client_params,
            target_params=target_params,
            name="duck_write_csv_documents",
            **kwargs,
        )
        if execute:
            ddb_csv.execute().materialize()
            sql_location = os.path.join(csv_location, "*.csv")
            self.table_name = table_name if not None else "data"
            if bool(glob.glob(sql_location)):
                client = duckdb.connect(":default:") if db_url is None else duckdb.connect(db_url)
                client.sql(f"CREATE TABLE {self.table_name} AS SELECT * FROM read_csv('{sql_location}')")
                # Flush out the csv files if not persisted
                if not csv_directory_location:
                    try:
                        for root, _, files in os.walk(csv_location):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    os.unlink(file_path)
                                except Exception as e:
                                    print(f"Error deleting {file_path}: {e}")
                    except Exception as e:
                        print(f"Error deleting files in {csv_location}: {e}")
            else:
                print(f"No files in directory matching the pattern in {sql_location}")
            return None
        else:
            from sycamore.docset import DocSet
            from sycamore.writers.duckdb_writer import DuckDB_Writer

            if db_url is None or db_url == ":default:":
                raise ValueError(
                    """Database cannot be run in-memory when not executed immediately. 
                    Please specify a persistent database location"""
                )
            ddb_writer = DuckDB_Writer(
                ddb_csv,
                csv_location=csv_location,
                table_name=table_name,
                csv_directory_location=csv_directory_location,
                db_url=db_url,
                name="duck_write_documents",
                **kwargs,
            )
            return DocSet(self.context, ddb_writer)

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
