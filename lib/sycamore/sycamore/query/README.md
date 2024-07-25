# Sycamore Query

This directory contains tools for using Sycamore to query unstructured data sources.
This code uses LLMs to take natural language queries and generate a query plan that runs on
Sycamore.

## Creating an index

The next step is to populate an index with some data. Be sure to have OpenSearch
running locally on port 9200. You can do this with:

```bash
$ cd .. && docker compose up
```

You can run the script `query/examples/ntsb_index_script.py`, which will populate a Sycamore
index from a set of NTSB incident reports. You can run it with:

```bash
$ cd lib/sycamore && poetry run python sycamore/query/examples/ntsb_index_script.py
```

## Using the web interface

There is a simple web app you can use to interact with Sycamore Query. You can run it with:

```bash
$ cd apps/query-ui && poetry run python -m streamlit run ./queryui/queryui.py
```

You can use this to issue a query against an index in the locally-running OpenSearch cluster.

## Using the command-line interface

There is also a CLI client in `query/client.py`:

```bash
$ poetry run ./libs/sycamore/sycamore/query/client.py "How many incident reports are there?"
```

This should print out the LLM-generated query plan, run the query, and give back a final
answer such as:

```
Result:

There are 73 incident reports in the database. This number is based on the latest data entry
and reflects the total count of all recorded incidents.
```

Run `poetry run ./libs/sycamore/sycamore/query/client.py --help` for more options.

## Using a Jupyter notebook

You can also use a Jupyter notebook to issue Sycamore queries. The notebook in `notebooks/query-demo.ipynb`
gives a good example.
