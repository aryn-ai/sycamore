#!/usr/bin/env python

# This script will populate a local OpenSearch index, using Sycamore, with data
# from NTSB incident reports.
# Run with poetry run python examples/query/ntsb_loader_materialized.py [--delete]

import argparse

import sycamore
import os
from opensearchpy import OpenSearch

argparser = argparse.ArgumentParser(prog="ntsb_loader")
argparser.add_argument("--delete", action="store_true")
argparser.add_argument("--oshost", default=None)
argparser.add_argument("--osport", default=9200)
args = argparser.parse_args()

# The S3 location of the raw NTSB data.
SOURCE_DATA_PATH = "s3://aryn-public/ntsb/"

# The OpenSearch index name to populate.
INDEX = "const_ntsb"

if args.oshost is not None:
    opensearch_host = args.oshost
elif os.path.exists("/.dockerenv"):
    opensearch_host = "opensearch"
    print("Assuming we are in a Sycamore Jupyter container, using opensearch for OpenSearch host")
else:
    opensearch_host = "localhost"
    print("Assuming we are running outside of a container, using localhost for OpenSearch host")

opensearch_port = args.osport

os_client_args = {
    "hosts": [{"host": opensearch_host, "port": opensearch_port}],
    "http_compress": True,
    "http_auth": ("admin", "admin"),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120,
}

os_client = OpenSearch(**os_client_args)  # type: ignore
if os_client.indices.exists(index=INDEX):
    if args.delete:
        print(f"Index {INDEX} already exists, deleting as requested")
        os_client.indices.delete(index=INDEX)
    else:
        raise Exception(f"Index {INDEX} already exists. Run with --delete to delete it.")

index_settings = {
    "body": {
        "settings": {"index.knn": True, "number_of_shards": 5, "number_of_replicas": 1},
        "mappings": {
            "properties": {
                "embedding": {
                    "dimension": 384,
                    "method": {
                        "engine": "faiss",
                        "space_type": "l2",
                        "name": "hnsw",
                        "parameters": {},
                    },
                    "type": "knn_vector",
                }
            }
        },
    }
}


context = sycamore.init()
_ = (
    context.read.materialize("s3://aryn-public/materialize/examples/luna/ntsb_loader_2024-08-29")
    .write.opensearch(
        os_client_args=os_client_args,
        index_name=INDEX,
        index_settings=index_settings,
    )
)
