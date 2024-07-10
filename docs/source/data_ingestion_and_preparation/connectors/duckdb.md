# DuckDB

[DuckDB](https://duckdb.org/) is an in-process analytical database that uses a SQL dialect and can read data from CSV, Parquet and JSON file formats. It requires no external dependencies and has the ability to run in-process on its host application. This simplicity in deployment makes it a good choice for lightweight operations when using Sycamore for development work or with low data loads.

## Configuration for DuckDB

*Please look at the [DuckDB API documentation](https://duckdb.org/docs/) for in-depth background on the following. We specify the essential portions for creating a new database outside of Sycamore below.*

DuckDB databases can either be persistent or in-memory. However, in-memory databases are currently incompatible with Sycamore's parallel processing, and only persistent databases are supported. To set up a new DuckDB database using Python, install the *duckdb* python package and run the following code:

```
import duckdb
duckdb_database = duckdb.connect(DUCKDB_FILE_PATH)
```

One can now write tables to the DuckDB database using the `duckdb_database` client and conduct queries from there outside of Sycamore. Note this is only required if a client wishes to conduct operations external to Sycamore on the database.

## Writing to DuckDB

To write a Docset to a DuckDB database table from Sycamore, use the DocSet `.write()` function. The DuckDB writer takes in the following arguments:

- db_url: The DuckDB database file path location. If there isn't a database already at this location, one will be created. Since only persistent databases are supported, only valid file paths are supported, with an error thrown otherwise. Keeping the database in memory (by use of an `:memory:` tag) will also throw an error. The default value is set to `tmp.db`.
- table_name: The chosen table for the documents. Note that if the table already exists, its schema will be validated to ensure writes can happen successfully. The default value is set to `default_table`.
- batch_size: Specifies the file batch size (multiplied by 1024) while entering entries into DuckDB. The default value is set to `1000`.
- schema: Specifies the schema of the table to enter entries into. Note that the entries must be compatible with the underlying PyArrow representation otherwise an error will be thrown. The default value is given below:
```
    schema: Optional[Dict[str, str]] = field(
        default_factory=lambda: {
            "doc_id": "VARCHAR",
            "embeddings": "DOUBLE[]",
            "properties": "MAP(VARCHAR, VARCHAR)",
            "text_representation": "VARCHAR",
            "bbox": "DOUBLE[]",
            "shingles": "BIGINT[]",
            "type": "VARCHAR",
        }
    )
```
- execute: Whether to execute the write immediately. The default value is set to `True`.

To use the writer, call write at the end of a Sycamore pipeline as done below:

```
ds.write.duckdb(table_name=table_name, db_url=db_url)
```

Note that the writer forces execution of all transforms before it, so would normally come at the end of a Sycamore pipeline. More information can be found in the {doc}`API documentation </APIs/data_preparation/docsetwriter>`.

## Reading from DuckDB

Reading from the DuckDB takes in only the `db_url` and `table_name` arguments, with the same specification and defaults as before. To read from a DuckDB database into a Sycamore DocSet, use the following code:

```
ctx = sycamore.init()
ctx.read.duckdb(db_url=db_url, table_name=table_name)
```

More information can be found in the {doc}`API documentation </APIs/data_preparation/docsetreader>`.
