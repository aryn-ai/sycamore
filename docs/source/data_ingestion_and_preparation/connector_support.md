# Writing Docsets to External Data Stores

Sycamore allows the ability to write Docsets out to external databases. This is needed to ensure the value gained from the ETL transfroms run in Sycamore can be in a usable format. Currently, Pinecone, Weaviate, Elasticsearch, DuckDB, and Opensearch are supported as external databases. Users can access this from a unified write interface, from where they can specify their databases and the respective connection and write arguments.

Further information for each supported database and its relevant documentation is given below:

* [DuckDB](./connectors/duckdb.md)
