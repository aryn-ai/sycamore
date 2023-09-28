![SycamoreLogoFinal.svg](https://raw.githubusercontent.com/aryn-ai/sycamore/main/docs/source/images/sycamore_logo.svg)

[![PyPI](https://img.shields.io/pypi/v/sycamore-ai)](https://pypi.org/project/sycamore-ai/)
[![Slack](https://img.shields.io/badge/slack-sycamore-brightgreen.svg?logo=slack)](https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg)
[![Docs](https://readthedocs.org/projects/sycamore/badge/?version=stable)](https://sycamore.readthedocs.io/en/stable/?badge=stable)
![License](https://img.shields.io/github/license/aryn-ai/sycamore)

Sycamore is a semantic data preparation system that makes it easy to transform and enrich your unstructured data and prepare it for search applications. It introduces a novel set-based abstraction that makes processing a large document collection as easy as reading a single document, and it comes with a scalable distributed runtime that makes it easy to go from prototype to production.

## Features

- Support for a variety of unstructured document formats, starting with PDF and HTML. More formats coming soon!
- LLM-enabled entity extraction to automatically pull out semantically meaningful information from your documents with just a few examples.
- Built-in data structures and transforms to make it easy to process large document collections. Sycamore is built around a data structure called the `DocSet` that represents a collection of unstructured documents, and supports transforms for chunking, manipulating, and augmenting these documents.
- Easily embed your data using a variety of popular embedding models. Sycamore will automatically batch records and leverage GPUs where appropriate.
- Scale your processing workloads from your laptop to the cloud without changing your application code. Sycamore is built on [Ray](https://ray.io), a distributed compute framework that can scale to hundreds of nodes.

## Resources

- PyPi: [https://pypi.org/project/sycamore-ai/](https://pypi.org/project/sycamore-ai/)
- Documentation: [https://sycamore.readthedocs.io](https://sycamore.readthedocs.io)
- Slack: [https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg](https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg)
- Aryn Docs: [https://docs.aryn.ai](https://docs.aryn.ai) Instructions for setting up an end-to-end conversational search application with Sycamore and OpenSearch.

## Installation

Sycamore currently runs on Python 3.9+ for Linux and Mac OS. To install, run

```bash
pip install sycamore-ai
```

For certain PDF processing operations, you also need to install `poppler`, which you can do with the OS-native package manager of your choice. For example, the command for Homebrew on Mac OS is

```bash
brew install poppler
```

## Getting Started

The following shows a simple Sycamore script to read a collection of PDFs, partition them, compute vector embeddings, and load them into a local OpenSearch cluster. This script currently expects that you configured OpenSearch locally as described in the [OpenSearch Docker documentation](https://opensearch.org/docs/latest/install-and-configure/install-opensearch/docker/#run-opensearch-in-a-docker-container). You should adjust based on your setup.

See our [documentation](https://sycamore.readthedocs.io) for lots more information and examples.

```python
# Import and initialize the Sycamore library.
import sycamore
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder

context = sycamore.init()

# Read a collection of PDF documents into a DocSet.
doc_set = context.read.binary(paths=["/path/to/pdfs/"], binary_format="pdf")

# Segment the pdfs using the Unstructured partitioner.
partitioned_doc_set = doc_set.partition(partitioner=UnstructuredPdfPartitioner())

# Compute vector embeddings for the individual components of each document.
embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
embedded_doc_set = partitioned_doc_set.explode() \
                                      .embed(embedder)

# Write the embedded documents to a local OpenSearch index.
os_client_args = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "use_ssl":True,
    "verify_certs":False,
    "http_auth":("admin", "admin")
}
embedded_doc_set.write.opensearch(os_client_args, "my_index_name")
```

## Contributing

Check out our [Contributing Guide](https://github.com/aryn-ai/sycamore/blob/main/CONTRIBUTING.md) for more information about how to contribute to Sycamore and set up your environment for development.
