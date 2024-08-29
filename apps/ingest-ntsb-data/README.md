# NTSB data ingestion pipeline

This is a simple Sycamore app to pull NTSB data from an S3 bucket and index it in OpenSearch.

First, run `poetry install` to install all dependencies.

Be sure OpenSearch is running locally on port 9200.

Be sure you have an Aryn API key set in your environment as `ARYN_API_KEY`,
and an OpenAI API key set as `OPENAI_API_KEY`.

Run with:

```bash
$ poetry run ingest_ntsb_data/main.py
```