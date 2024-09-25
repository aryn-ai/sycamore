# Sycamore Query Web UI

This is a simple web-based UI for Sycamore Query.

## Setup

First, run `poetry install` to install all dependencies.

Be sure to have OpenSearch running locally on port 9200. You can do this with:

```bash
$ cd ../.. && docker compose up
```

## Using the web interface

Run it with:

```bash
$ poetry run python queryui/main.py
```

You can use this to issue a query against an index in the locally-running OpenSearch cluster.

