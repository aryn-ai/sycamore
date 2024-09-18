from typing import Optional, Union, Callable, Dict
from pathlib import Path

from pandas import DataFrame
from pyarrow import Table
from pyarrow.filesystem import FileSystem

from sycamore.context import context_params
from sycamore.plan_nodes import Node
from sycamore import Context, DocSet
from sycamore.data import Document
from sycamore.connectors.file import ArrowScan, BinaryScan, DocScan, PandasScan, JsonScan, JsonDocumentScan
from sycamore.connectors.file.file_scan import FileMetadataProvider
from sycamore.utils.import_utils import requires_modules


class DocSetReader:
    """
    Contains interfaces for reading from external storage systems.

    Users should not instantiate this class directly, but instead access an instance using
    :meth:`sycamore.context.read`
    """

    def __init__(self, context: Context, plan: Optional[Node] = None):
        self._context = context
        self.plan = plan

    def materialize(self, path: Union[Path, str], **kwargs) -> DocSet:
        """Read a docset via materialization.

        Semantics are a subset of the allowed options for DocSet.materialize.
        source_mode is always IF_PRESENT.  Complex path specifications are disallowed
        since reading from materialization requires default options."""

        from sycamore.materialize import Materialize, MaterializeSourceMode

        m = Materialize(child=None, context=self._context, path=path, source_mode=MaterializeSourceMode.USE_STORED)
        return DocSet(self._context, m)

    def binary(
        self,
        paths: Union[str, list[str]],
        binary_format: str,
        parallelism: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        **kwargs,
    ) -> DocSet:
        """
        Reads the contents of Binary Files into a DocSet

        Args:
            paths: Paths to the Binary file
            binary_format:  Binary file format to read from
            parallelism: (Optional) Override the number of output blocks from all read tasks. Defaults to
                -1 if not specified
            filesystem: (Optional) The PyArrow filesystem to read from. By default is selected based on the
                scheme of the paths passed in
            kwargs: (Optional) Arguments to passed into the underlying execution engine

        Example:
        The following shows how read pdfs from an S3 file path.

        .. code-block:: python

            paths = "s3://aryn-public/sort-benchmark/pdf/"
            # Initializng sycamore which also initializes Ray underneath
            context = sycamore.init()
            # Creating a DocSet
            docset = context.read.binary(paths, parallelism=1, binary_format="pdf")
        """
        scan = BinaryScan(
            paths,
            binary_format=binary_format,
            parallelism=parallelism,
            filesystem=filesystem,
            metadata_provider=metadata_provider,
            **kwargs,
        )
        return DocSet(self._context, scan)

    # TODO: Support including the metadata attributes in the manifest file directly
    def manifest(
        self,
        metadata_provider: FileMetadataProvider,
        binary_format: str,
        parallelism: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        **kwargs,
    ) -> DocSet:
        """
        Reads the contents of Binary Files into a DocSet using the Metadata manifest as their paths

        Args:
            metadata_provider: Metadata provider for each file, with the manifest being used as the paths to read from
            binary_format:  Binary file format to read from
            parallelism: (Optional) Override the number of output blocks from all read tasks. Defaults to
                -1 if not specified
            filesystem: (Optional) The PyArrow filesystem to read from. By default is selected based on the scheme
                of the paths passed in
            kwargs: (Optional) Arguments to passed into the underlying execution engine

        Example:
        The following shows how read a JSON manifest file into a Sycamore DocSet.

        .. code-block:: python

            base_path = str("resources/data/htmls/")
            remote_url = "https://en.wikipedia.org/wiki/Binary_search_algorithm"
            indexed_at = "2023-10-04"
            manifest = {base_path + "/wikipedia_binary_search.html": {"remote_url": remote_url,
            "indexed_at": indexed_at}}
            manifest_loc = str(f"TMP-PATH/manifest.json")

            with open(manifest_loc, "w") as file:
                json.dump(manifest, file)
            context = sycamore.init()
            docset = context.read.manifest(JsonManifestMetadataProvider(manifest_loc), binary_format="html")

        """
        paths = metadata_provider.get_paths()
        scan = BinaryScan(
            paths,
            binary_format=binary_format,
            parallelism=parallelism,
            filesystem=filesystem,
            metadata_provider=metadata_provider,
            **kwargs,
        )
        return DocSet(self._context, scan)

    def json(
        self,
        paths: Union[str, list[str]],
        properties: Optional[Union[str, list[str]]] = None,
        metadata_provider: Optional[FileMetadataProvider] = None,
        document_body_field: Optional[str] = None,
        doc_extractor: Optional[Callable] = None,
        **kwargs,
    ) -> DocSet:
        """
         Reads the contents of JSON Documents into a DocSet

         Args:
             paths: Paths to JSON documents to read into a DocSet
             properties: (Optional) Properties to be extracted into the DocSet
             metadata_provider: (Optional) Metadata provider for each file
                (will be added to the Document's metadata)
             document_body_field: (Optional) Document Body Field specification.
                Will use the entire json output otherwise.
             doc_extractor: (Optional) Custom function to convert the JSON document to a Sycamore Document
             kwargs: (Optional) Arguments to passed into the underlying execution engine

        Example:
         The following shows how read a JSON file into a Sycamore DocSet.

         .. code-block:: python

            docset = context.read.json("s3://bucket/prefix/json")
        """
        json_scan = JsonScan(
            paths,
            properties=properties,
            metadata_provider=metadata_provider,
            document_body_field=document_body_field,
            doc_extractor=doc_extractor,
            **kwargs,
        )
        return DocSet(self._context, json_scan)

    def json_document(self, paths: Union[str, list[str]], **kwargs) -> DocSet:
        """
        Reads the contents of JSONL Documents into a DocSet

        Args:
            paths: Paths to JSONL documents to read into a DocSet
        """
        scan = JsonDocumentScan(paths, **kwargs)
        return DocSet(self._context, scan)

    def arrow(self, tables: Union[Table, bytes, list[Union[Table, bytes]]]) -> DocSet:
        """
        Reads the contents of PyArrow Tables into a DocSet

        Args:
            tables: PyArrow Tables to read into a DocSet
        """
        scan = ArrowScan(tables)
        return DocSet(self._context, scan)

    def document(self, docs: list[Document], **kwargs) -> DocSet:
        """
        Reads the contents of Sycamore Documents into a DocSet

        Args:
            docs: Sycamore Documents to read into a DocSet
        """
        scan = DocScan(docs, **kwargs)
        return DocSet(self._context, scan)

    def pandas(self, dfs: Union[DataFrame, list[DataFrame]]) -> DocSet:
        """
        Reads the contents of Pandas Dataframes into a DocSet

        Args:
            dfs: Pandas DataFrames to read into a DocSet
        """
        scan = PandasScan(dfs)
        return DocSet(self._context, scan)

    @requires_modules("opensearchpy", extra="opensearch")
    @context_params
    def opensearch(self, os_client_args: dict, index_name: str, query: Optional[Dict] = None, **kwargs) -> DocSet:
        """
        Reads the content of an OpenSearch index into a DocSet.

        Args:
            os_client_args: Keyword parameters that are passed to the opensearch-py OpenSearch client constructor.
                See more information at https://opensearch.org/docs/latest/clients/python-low-level/
            index_name: Index name to write to in the OpenSearch instance
            query: (Optional) Query to perform on the index. Note that this must be specified in the OpenSearch
                Query DSL as a dictionary. Otherwise, it defaults to a full scan of the table. See more information at
                https://opensearch.org/docs/latest/query-dsl/
        Example:
            The following shows how to write to data into a OpenSearch Index, and read it back into a DocSet.

            .. code-block:: python

                INDEX = "test_opensearch_read"

                OS_CLIENT_ARGS = {
                    "hosts": [{"host": "localhost", "port": 9200}],
                    "http_compress": True,
                    "http_auth": ("admin", "admin"),
                    "use_ssl": True,
                    "verify_certs": False,
                    "ssl_assert_hostname": False,
                    "ssl_show_warn": False,
                    "timeout": 120,
                }
                path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
                context = sycamore.init()
                original_docs = (
                    context.read.binary(path, binary_format="pdf")
                    .partition(partitioner=UnstructuredPdfPartitioner())
                    .explode()
                    .write.opensearch(
                        os_client_args=OS_CLIENT_ARGS, index_name=INDEX, execute=False
                    )
                    .take_all()
                )

                retrieved_docs = context.read.opensearch(
                    os_client_args=OS_CLIENT_ARGS, index_name=INDEX
                )
                target_doc_id = original_docs[-1].doc_id if original_docs[-1].doc_id else ""
                query = {"query": {"term": {"_id": target_doc_id}}}
                query_docs = context.read.opensearch(
                    os_client_args=OS_CLIENT_ARGS, index_name=INDEX, query=query
                )
        """
        from sycamore.connectors.opensearch import (
            OpenSearchReader,
            OpenSearchReaderClientParams,
            OpenSearchReaderQueryParams,
        )

        client_params = OpenSearchReaderClientParams(os_client_args=os_client_args)
        query_params = (
            OpenSearchReaderQueryParams(index_name=index_name, query=query)
            if query is not None
            else OpenSearchReaderQueryParams(index_name=index_name)
        )
        osr = OpenSearchReader(client_params=client_params, query_params=query_params)
        return DocSet(self._context, osr)

    @requires_modules("duckdb", extra="duckdb")
    def duckdb(
        self, db_url: str, table_name: str, create_hnsw_table: Optional[str] = None, query: Optional[str] = None
    ) -> DocSet:
        """
        Reads the content of a DuckDB database index into a DocSet.

        Args:
            db_url: The URL of the DuckDB database.
            table_name: The table name to read the data from
            create_hnsw_table: (Optional) SQL query to add an HNSW index to the DuckDB before conducting a read.
                More information is available at https://duckdb.org/docs/extensions/vss
            query: (Optional) SQL query to read from the table. If not specified, the read will perform
                a full scan of the table

        Example:
            The following shows how to write to data into a DuckDB database and get it back as a DocSet.

            .. code-block:: python

                table_name = "duckdb_table"
                db_url = "tmp_read.db"
                paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                tokenizer = HuggingFaceTokenizer(model_name)

                ctx = sycamore.init()

                docs = (
                    ctx.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=UnstructuredPdfPartitioner())
                    .regex_replace(COALESCE_WHITESPACE)
                    .mark_bbox_preset(tokenizer=tokenizer)
                    .merge(merger=MarkedMerger())
                    .spread_properties(["path"])
                    .split_elements(tokenizer=tokenizer, max_tokens=512)
                    .explode()
                    .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
                    .sketch(window=17)
                    .take_all()
                )
                ctx.read.document(docs).write.duckdb(db_url=db_url, table_name=table_name, dimensions=384)
                target_doc_id = docs[-1].doc_id if docs[-1].doc_id else ""
                out_docs = ctx.read.duckdb(db_url=db_url, table_name=table_name).take_all()
                query = f"SELECT * from {table_name} WHERE doc_id == '{target_doc_id}'"
                query_docs = ctx.read.duckdb(db_url=db_url, table_name=table_name, query=query).take_all()
        """
        from sycamore.connectors.duckdb import DuckDBReader, DuckDBReaderClientParams, DuckDBReaderQueryParams

        client_params = DuckDBReaderClientParams(db_url=db_url)
        query_params = DuckDBReaderQueryParams(table_name=table_name, query=query, create_hnsw_table=create_hnsw_table)
        ddbr = DuckDBReader(client_params=client_params, query_params=query_params)
        return DocSet(self._context, ddbr)

    @requires_modules("pinecone", extra="pinecone")
    def pinecone(self, index_name: str, api_key: str, namespace: str = "", query: Optional[Dict] = None) -> DocSet:
        """
        Reads the content of a Pinecone database index into a DocSet.

        Args:
            index_name: Name of the pinecone index to ingest into
            api_key: Pinecone service API Key. Defaults to None (will use the environment
                    variable PINECONE_API_KEY).
            namespace: Namespace withing the pinecone index to ingest into. See
                https://docs.pinecone.io/guides/indexes/use-namespaces for more information.
                Defaults to "", which is the default namespace
            query: (Optional) Dictionary of parameters to pass into the pinecone `index.query()` method.
                If not specified, will default to a full scan of the index.
                See more information at https://docs.pinecone.io/guides/data/query-data

        Example:
            The following shows how to write to data into a Pinecone index and read it back as a DocSet.

            .. code-block:: python

                spec = ServerlessSpec(cloud="aws", region="us-east-1")
                index_name = "test-index-read"
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                namespace = f"{generate_random_string().lower()}"
                paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")
                api_key = os.environ.get("PINECONE_API_KEY", "")
                assert (
                    api_key is not None
                ), "Missing api key: either provide it as an argument or set the PINECONE_API_KEY env variable."

                pc = PineconeGRPC(api_key=api_key)

                tokenizer = HuggingFaceTokenizer(model_name)

                ctx = sycamore.init()

                docs = (
                    ctx.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=UnstructuredPdfPartitioner())
                    .regex_replace(COALESCE_WHITESPACE)
                    .mark_bbox_preset(tokenizer=tokenizer)
                    .merge(merger=MarkedMerger())
                    .spread_properties(["path"])
                    .split_elements(tokenizer=tokenizer, max_tokens=512)
                    .explode()
                    .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
                    .sketch(window=17)
                    .take_all()
                )
                ctx.read.document(docs).write.pinecone(index_name=index_name, dimensions=384,
                namespace=namespace, index_spec=spec)
                target_doc_id = docs[-1].doc_id if docs[-1].doc_id and docs[0].doc_id else ""
                if len(target_doc_id) > 0:
                    target_doc_id = f"{docs[-1].parent_id}#{target_doc_id}" if docs[-1].parent_id else target_doc_id
                wait_for_write_completion(client=pc, index_name=index_name, namespace=namespace, doc_id=target_doc_id)
                out_docs = ctx.read.pinecone(index_name=index_name, api_key=api_key, namespace=namespace).take_all()
                query_params = {"namespace": namespace, "id": target_doc_id, "top_k": 1, "include_values": True}
                query_docs = ctx.read.pinecone(
                    index_name=index_name, api_key=api_key, query=query_params, namespace=namespace
                ).take_all()
        """
        from sycamore.connectors.pinecone import PineconeReader, PineconeReaderClientParams, PineconeReaderQueryParams

        client_params = PineconeReaderClientParams(api_key=api_key)
        query_params = PineconeReaderQueryParams(index_name=index_name, query=query, namespace=namespace)
        pr = PineconeReader(client_params=client_params, query_params=query_params)
        return DocSet(self._context, pr)

    @requires_modules("elasticsearch", extra="elasticsearch")
    def elasticsearch(
        self, url: str, index_name: str, es_client_args: dict = {}, query: Optional[Dict] = None, **kwargs
    ) -> DocSet:
        """
        Reads the content of an Elasticsearch index into a DocSet.

        Args:
            url: Connection endpoint for the Elasticsearch instance. Note that this must be paired with the
                necessary client arguments below
            index_name: Index name to write to in the Elasticsearch instance
            es_client_args: Authentication arguments to be specified (if needed). See more information at
                https://elasticsearch-py.readthedocs.io/en/v8.14.0/api/elasticsearch.html
            query: (Optional) Query to perform on the index. Note that this must be specified in the Elasticsearch
                Query DSL as a dictionary. Otherwise, it defaults to a full scan of the table.
                See more information at
                https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html
            kwargs: (Optional) Parameters to pass in to the underlying Elasticsearch search query.
                See more information at
                https://elasticsearch-py.readthedocs.io/en/v8.14.0/api/elasticsearch.html#elasticsearch.Elasticsearch.search
        Example:
            The following shows how to write to data into a Elasticsearch Index, and read it back into a DocSet.

            .. code-block:: python

                url = "http://localhost:9201"
                index_name = "test_index-read"
                wait_for_completion = "wait_for"
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")

                OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
                tokenizer = HuggingFaceTokenizer(model_name)

                ctx = sycamore.init()

                docs = (
                    ctx.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=UnstructuredPdfPartitioner())
                    .regex_replace(COALESCE_WHITESPACE)
                    .mark_bbox_preset(tokenizer=tokenizer)
                    .merge(merger=MarkedMerger())
                    .spread_properties(["path"])
                    .split_elements(tokenizer=tokenizer, max_tokens=512)
                    .explode()
                    .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
                    .sketch(window=17)
                    .take_all()
                )
                ctx.read.document(docs).write.elasticsearch(url=url, index_name=index_name,
                wait_for_completion=wait_for_completion)
                target_doc_id = docs[-1].doc_id if docs[-1].doc_id else ""
                out_docs = ctx.read.elasticsearch(url=url, index_name=index_name).take_all()
                query_params = {"term": {"_id": target_doc_id}}
                query_docs = ctx.read.elasticsearch(url=url, index_name=index_name, query=query_params).take_all()
        """
        from sycamore.connectors.elasticsearch import (
            ElasticsearchReader,
            ElasticsearchReaderClientParams,
            ElasticsearchReaderQueryParams,
        )

        client_params = ElasticsearchReaderClientParams(url=url, es_client_args=es_client_args)
        query_params = (
            ElasticsearchReaderQueryParams(index_name=index_name, query=query, kwargs=kwargs)
            if query is not None
            else ElasticsearchReaderQueryParams(index_name=index_name, kwargs=kwargs)
        )

        esr = ElasticsearchReader(client_params=client_params, query_params=query_params)
        return DocSet(self._context, esr)

    @requires_modules("weaviate", extra="weaviate")
    def weaviate(self, wv_client_args: dict, collection_name: str, **kwargs) -> DocSet:
        """
        Reads the content of a Weaviate collection into a DocSet.

        Args:
            wv_client_args: Keyword parameters that are passed to the weaviate client constructor. See more information
                at
                https://weaviate.io/developers/weaviate/client-libraries/python#python-client-v4-explicit-connection
            collection_name: The name of the Weaviate collection into which to load this DocSet.
            kwargs: (Optional) Search queries to pass into Weaviate. Note each keyword method argument
                must have its parameters specified as a dictionary. Will default to a full scan if not specified.
                See more information below  and at https://weaviate.io/developers/weaviate/search
        Example:
            The following shows how to write to data into a Weaviate collection, and read it back into a DocSet.

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

                docs = (
                        ctx.read.binary(paths, binary_format="pdf")
                        .partition(partitioner=UnstructuredPdfPartitioner())
                        .regex_replace(COALESCE_WHITESPACE)
                        .mark_bbox_preset(tokenizer=tokenizer)
                        .merge(merger=MarkedMerger())
                        .spread_properties(["path"])
                        .split_elements(tokenizer=tokenizer, max_tokens=512)
                        .explode()
                        .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
                        .sketch(window=17)
                        .take_all()
                    )
                ctx.read.document(docs).write.weaviate(
                    wv_client_args=wv_client_args, collection_name=collection,
                    collection_config=collection_config_params
                )
                out_docs = ctx.read.weaviate(wv_client_args=wv_client_args, collection_name=collection).take_all()
                target_doc_id = docs[-1].doc_id if docs[-1].doc_id else ""
                fetch_object_dict = {"filters": Filter.by_id().equal(target_doc_id)}
                query_docs = ctx.read.weaviate(
                    wv_client_args=wv_client_args, collection_name=collection, fetch_objects=fetch_object_dict
                ).take_all()
        """
        from sycamore.connectors.weaviate import (
            WeaviateReader,
            WeaviateReaderClientParams,
            WeaviateReaderQueryParams,
        )

        client_params = WeaviateReaderClientParams(**wv_client_args)
        query_params = WeaviateReaderQueryParams(collection_name=collection_name, query_kwargs=kwargs)
        wr = WeaviateReader(client_params=client_params, query_params=query_params)
        return DocSet(self._context, wr)

    @requires_modules("qdrant_client", extra="qdrant")
    def qdrant(self, client_params: dict, query_params: dict) -> DocSet:
        """
        Reads the contents of a Qdrant collection into a DocSet.

        Args:
            client_params: Parameters that are passed to the Qdrant client constructor.
            See more information at
            https://python-client.qdrant.tech/qdrant_client.qdrant_client
            query_params: Parameters that are passed into the qdrant_client.QdrantClient.query_points method.
            See more information at
            https://python-client.qdrant.tech/_modules/qdrant_client/qdrant_client#QdrantClient.query_points
        """
        from sycamore.connectors.qdrant import (
            QdrantReader,
            QdrantReaderClientParams,
            QdrantReaderQueryParams,
        )

        client_params = QdrantReaderClientParams(**client_params)
        query_params = QdrantReaderQueryParams(query_params=query_params)
        wr = QdrantReader(client_params=client_params, query_params=query_params)
        return DocSet(self._context, wr)
