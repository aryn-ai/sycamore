![SycamoreLogoFinal.svg](https://raw.githubusercontent.com/aryn-ai/sycamore/main/docs/source/images/sycamore_logo.svg)

[![PyPI](https://img.shields.io/pypi/v/sycamore-ai)](https://pypi.org/project/sycamore-ai/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sycamore-ai)](https://pypi.org/project/sycamore-ai/)
[![Slack](https://img.shields.io/badge/slack-sycamore-brightgreen.svg?logo=slack)](https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg)
[![Docs](https://readthedocs.org/projects/sycamore/badge/?version=stable)](https://sycamore.readthedocs.io/en/stable/?badge=stable)
![License](https://img.shields.io/github/license/aryn-ai/sycamore)

Sycamore is an AI-powered document segmenation and enrichment framework for LLMs, RAG, and unstructured analytics. It makes it easy to reliably load your vector databases and hybrid search engines with higher quality data. Sycamore can process and prepare a wide range of document types including reports, presentations, transcripts, manuals, and more. It can analyze and chunk complex documents such as PDFs and images with embedded tables, figures, graphs, and other infographics. Sycamore provides an advanced document partitioner for high fidelity segmentation and chunking, and can also integrate with your choice of AI models for LLM-powered UDFs, vector embeddings, and other data transformations. It supports a variety of engines such as OpenSearch and Weaviate and reads from Amazon S3.

The Sycamore framework is built around a scalable and robust abstraction for document processing called a DocSet, and includes powerful high-level transformations in Python for data processing, metadata extraction, enrichment, and cleaning. DocSets also encapsulate scalable data processing techniques removing the undifferentiated heavy lifting of reliably loading chunks. DocSets' functional programming approach allows you to rapidly customize and experiment with your chunking for better quality RAG results.

![Untitled](docs/source/images/SycamoreDataflowDiagramv1.png)

## Features

- State-of-the art vision AI model for segmentation and preserving the semantic structure of documents
- DocSet abstraction to scalably and reliably transform and manipulate unstructured documents
- High-quality table extraction, OCR, visual summarization, LLM-powered UDFs, and other performant Python data transforms
- Quickly create vector embeddings using your choice of AI model
- Helpful features like automatic data crawlers (Amazon S3 and HTTP), Jupyter notebook for writing and iterating on jobs, and an OpenSearch hybrid search and RAG engine for testing
- Scalable [Ray](https://github.com/ray-project/ray) backend

## Demo

[Hosted on Loom](https://www.loom.com/share/53e68b0eb5ab49948111a3fcf6286b7f?sid=8627ff2a-db36-46ef-9762-a01b37e20ced)

## Get Started

You can easily deploy Sycamore locally or on a virtual machine using Docker.

With Docker installed:

1.	Clone the Sycamore repo:

```git clone https://github.com/aryn-ai/sycamore```

2.	Set OpenAI Key:

```export OPENAI_API_KEY=YOUR-KEY```

3.	Go to:

```/sycamore```

4.	Launch Sycamore. Conatainers will be pulled from DockerHub:

```docker compose up --pull=always```

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
