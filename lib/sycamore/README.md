![SycamoreLogoFinal.svg](https://raw.githubusercontent.com/aryn-ai/sycamore/main/docs/source/images/sycamore_logo.svg)

[![PyPI](https://img.shields.io/pypi/v/sycamore-ai)](https://pypi.org/project/sycamore-ai/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sycamore-ai)](https://pypi.org/project/sycamore-ai/)
[![Slack](https://img.shields.io/badge/slack-sycamore-brightgreen.svg?logo=slack)](https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg)
[![Docs](https://readthedocs.org/projects/sycamore/badge/?version=stable)](https://sycamore.readthedocs.io/en/stable/?badge=stable)
![License](https://img.shields.io/github/license/aryn-ai/sycamore)

Sycamore is an open source, AI-powered document processing engine for ETL, RAG, LLM-based applications, and analytics on unstructured data. Sycamore can partition and enrich a wide range of document types including reports, presentations, transcripts, manuals, and more. It can analyze and chunk complex documents such as PDFs and images with embedded tables, figures, graphs, and other infographics. Check out an [example notebook](https://github.com/aryn-ai/sycamore/blob/main/notebooks/sycamore-tutorial-intermediate-etl.ipynb).

For processing documents, Sycamore leverages [Aryn DocParse](https://www.aryn.ai/post/announcing-the-aryn-partitioning-service) (formerly known as the Aryn Partitioning Service), a serverless, GPU-powered API for segmenting and labeling documents, doing OCR, extracting tables and images, and more. It levereages Aryn's state-of-the-art, [open source deep learning DETR AI model](https://huggingface.co/Aryn/deformable-detr-DocLayNet) trained on 80k+ enterprise documents, and it can lead to 6x more accurate data chunking and 2x improved recall on hybrid search or RAG when compared to alternate systems. You can [sign-up for free here](http://www.aryn.ai/get-started), or choose to run the Aryn Partitioner locally.

Aryn DocParse takes [documents](https://docs.aryn.ai/docparse/formats_supported) and returns the partitioned output in JSON, and you can use Sycamore for additional data extraction, enrichment, transforms, cleaning, and loading into downstream databases. You can choose the LLMs to use with these transforms.

Sycamore reliably loads your vector databases and hybrid search engines, including as OpenSearch, ElasticSearch, Pinecone, DuckDB and Weaviate, with higher quality data. 

The Sycamore framework is built around a scalable and robust abstraction for document processing called a DocSet, and includes powerful high-level transformations in Python for data processing, enrichment, and cleaning. DocSets also encapsulate scalable data processing techniques removing the undifferentiated heavy lifting of reliably loading chunks. DocSets' functional programming approach allows you to rapidly customize and experiment with your chunking for better quality RAG results.

![Untitled](docs/source/images/SycamoreDataflowDiagramv2.png)

## Features

- Integrated with [Aryn Aryn DocParse](https://docs.aryn.ai/quickstart), using a [state-of-the art vision AI model](https://huggingface.co/Aryn/deformable-detr-DocLayNet) for segmentation and preserving the semantic structure of documents
- DocSet abstraction to scalably and reliably transform and manipulate unstructured documents
- High-quality table extraction, OCR, visual summarization, LLM-powered UDFs, and other performant Python data transforms
- Quickly create vector embeddings using your choice of AI model
- Helpful features like automatic data crawlers (Amazon S3 and HTTP), Jupyter notebook for writing and iterating on jobs, and an OpenSearch hybrid search and RAG engine for testing
- Scalable [Ray](https://github.com/ray-project/ray) backend

## Demo

[Introduction to Aryn DocParse](https://www.aryn.ai/?name=ArynPartitioningService_Intro)

## Get Started

Sycamore currently runs on Linux and Mac OS. To install , run:

```pip install sycamore-ai```

Sycamore provides connectors to vector databases via Python extras. To install a connector, include it as an extra with your pip install. For example, 

```pip install sycamore-ai[duckdb]```

Supported connectors include `duckdb`, `elasticsearch`, `opensearch`, `pinecone`, and `weaviate`.

To use Aryn DocParse, [sign-up for free here](https://www.aryn.ai/get-started) and use the API key.

## Resources

- Documentation: https://sycamore.readthedocs.io
- Example notebook: https://github.com/aryn-ai/sycamore/blob/main/notebooks/sycamore-tutorial-intermediate-etl.ipynb
- Slack: https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg
- Data preparation libraries (PyPi): https://pypi.org/project/sycamore-ai/
- Contact us: info@aryn.ai

## Contributing

Check out our [Contributing Guide](https://github.com/aryn-ai/sycamore/blob/main/CONTRIBUTING.md) for more information about how to contribute to Sycamore and set up your environment for development.
