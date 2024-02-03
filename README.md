![SycamoreLogoFinal.svg](https://raw.githubusercontent.com/aryn-ai/sycamore/main/docs/source/images/sycamore_logo.svg)

[![PyPI](https://img.shields.io/pypi/v/sycamore-ai)](https://pypi.org/project/sycamore-ai/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sycamore-ai)](https://pypi.org/project/sycamore-ai/)
[![Slack](https://img.shields.io/badge/slack-sycamore-brightgreen.svg?logo=slack)](https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg)
[![Docs](https://readthedocs.org/projects/sycamore/badge/?version=stable)](https://sycamore.readthedocs.io/en/stable/?badge=stable)
![License](https://img.shields.io/github/license/aryn-ai/sycamore)

Sycamore is a conversational search and analytics platform for complex unstructured data, such as documents, presentations, transcripts, embedded tables, and internal knowledge repositories. It retrieves and synthesizes high-quality answers through bringing AI to data preparation, indexing, and retrieval. Sycamore makes it easy to prepare unstructured data for search and analytics, providing a toolkit for data cleaning, information extraction, enrichment, summarization, and generation of vector embeddings that encapsulate the semantics of data. Sycamore uses your choice of generative AI models to make these operations simple and effective, and it enables quick experimentation and iteration. Additionally, Sycamore uses OpenSearch for indexing, enabling hybrid (vector + keyword) search, retrieval-augmented generation (RAG) pipelining, filtering, analytical functions, conversational memory, and other features to improve information retrieval.

![Untitled](docs/source/images/SycamoreDiagram2.png)

## Features

- Natural language, conversational interface to ask complex questions on unstructured data. Includes citations to source passages and conversational memory.
- Includes a variety of query operations over unstructured data, including hybrid search, retrieval augmented generation (RAG), and analytical functions.
- Prepares and enriches complex unstructured data for search and analytics through advanced data segmentation, LLM-powered UDFs for data enrichment, performant data manipulation with Python, and vector embeddings using a variety of AI models.
- Helpful features like automatic data crawlers (Amazon S3 and HTTP) and Jupyter notebook support to create and iterate on data preparation scripts.
- Scalable, secure, and customizable OpenSearch backend for indexing and data retrieval.

## Demo

[Hosted on Loom](https://www.loom.com/share/53e68b0eb5ab49948111a3fcf6286b7f?sid=8627ff2a-db36-46ef-9762-a01b37e20ced)

## Get Started

You can easily deploy Sycamore locally or on a virtual machine using Docker.

With Docker installed:

1.	Clone the Sycamore repo: git clone https://github.com/aryn-ai/sycamore
2.	Set OpenAI Key

```export OPENAI_API_KEY=YOUR-KEY```

3.	Go to /sycamore/deployment/docker_compose
4.	Launch Sycamore. Conatainers will be pulled from DockerHub:

```Docker compose up --pull-always```

5.	The Sycamore demo query UI will be at localhost:3000

You can next choose to run a demo that [prepares and ingests data from the Sort Benchmark website](docs/source/welcome_to_sycamore/get_started.md#demo-ingest-and-query-sort-benchmark-dataset), [crawl data from a public website](docs/source/welcome_to_sycamore/get_started.md#demo-ingest-and-query-data-from-an-arbitrary-website), or write your own data preparation script.

For more info about Sycamore’s data ingestion and preparation feature set, visit the [Sycamore documentation](docs/source/data_ingestion_and_preparation/data_preparation_concepts.md).


## Resources

- Documentation: https://sycamore.readthedocs.io
- Slack: https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg
- Data preparation libraries (PyPi): https://pypi.org/project/sycamore-ai/
- Contact us: info@aryn.ai

## Contributing

Check out our [Contributing Guide](https://github.com/aryn-ai/sycamore/blob/main/CONTRIBUTING.md) for more information about how to contribute to Sycamore and set up your environment for development.
