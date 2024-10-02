# Sycamore Query Server

This is a simple server that provides a REST API wrapper to the Sycamore Query
functionality found in the `sycamore.query` package.

It currently assumes that OpenSearch is running locally. This code performs no authentication,
so any OpenSearch index available for local querying will be accessible through this server.

To build and run the server:
```bash
$ poetry install
$ poetry run fastapi dev queryserver/main.py
```
