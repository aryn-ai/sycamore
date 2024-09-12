Vector Database Ingestion
=============

Writing a DocSet to Target Data Stores
--------------------------------------

A final step in a Sycamore processing job is to load the data into a target database for use in your application. This could be a combination of a vector index and term-based index, and includes the enriched metadata from the job. Currently, Pinecone, Weaviate, Elasticsearch, DuckDB, and Opensearch are supported as target databases. Users can access this from a unified write interface, from where they can specify their databases and the respective connection and write arguments.

Further information for each supported database and its relevant documentation is given below:

.. toctree::
    :maxdepth: 1

    ./connectors/duckdb.md
    ./connectors/weaviate.md
    ./connectors/pinecone.md
    ./connectors/elasticsearch.md
    ./connectors/opensearch.md

