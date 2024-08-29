#!/usr/bin/env python

# This script will populate a local OpenSearch index, using Sycamore, with data
# from NTSB incident reports.

import argparse

import sycamore
from sycamore.data import Document
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.functions import HuggingFaceTokenizer
from sycamore.materialize import MaterializeSourceMode
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.transforms.extract_schema import (
    OpenAIPropertyExtractor,
)
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms.embed import SentenceTransformerEmbedder
import os
from dateutil import parser
from opensearchpy import OpenSearch


argparser = argparse.ArgumentParser(prog='ntsb_loader')
argparser.add_argument('--delete', action='store_true')
args = argparser.parse_args()

# The S3 location of the raw NTSB data.
SOURCE_DATA_PATH = "s3://aryn-public/ntsb/"

# The OpenSearch index name to populate.
INDEX = "const_ntsb"


def add_property_to_schema(doc: Document) -> Document:
    schema_json = {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "dateAndTime": {"type": "string"},
            "aircraft": {"type": "string"},
            "flightConductedUnder": {"type": "string"},
            "accidentNumber": {"type": "string"},
            "registration": {"type": "string"},
            "injuries": {"type": "string"},
            "aircraftDamage": {"type": "string"},
        },
    }
    doc.properties.update({"_schema": schema_json, "_schema_class": "Flight Accident Report"})
    return doc


def convert_timestamp(doc: Document) -> Document:
    if "dateAndTime" not in doc.properties["entity"] and "dateTime" not in doc.properties["entity"]:
        return doc
    raw_date: str = doc.properties["entity"].get("dateAndTime") or doc.properties["entity"].get("dateTime")
    raw_date = raw_date.replace("Local", "")
    parsed_date = parser.parse(raw_date, fuzzy=True)
    extracted_date = parsed_date.date()
    doc.properties["entity"]["day"] = extracted_date.isoformat()
    if parsed_date.utcoffset():
        doc.properties["entity"]["isoDateTime"] = parsed_date.isoformat()
    else:
        doc.properties["entity"]["isoDateTime"] = parsed_date.isoformat() + "Z"

    return doc


if os.path.exists("/.dockerenv"):
    opensearch_host = "opensearch"
    print("Assuming we are in a Sycamore Jupyter container, using opensearch for OpenSearch host")
else:
    opensearch_host = "localhost"
    print("Assuming we are running outside of a container, using localhost for OpenSearch host")


os_client_args = {
    "hosts": [{"host": opensearch_host, "port": 9200}],
    "http_compress": True,
    "http_auth": ("admin", "admin"),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120,
}

client = OpenSearch(**os_client_args)
if client.indices.exists(index=INDEX):
    if args.delete:
        print(f"Index {INDEX} already exists, deleting as requested")
        client.indices.delete(index=INDEX)
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
docset = context.read.binary(SOURCE_DATA_PATH, binary_format="pdf")
tokenizer = HuggingFaceTokenizer("thenlper/gte-small")
llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
client = OpenAI(OpenAIModels.GPT_4O.value)

# partitioning docset
partitioned_docset = (
    docset.partition(partitioner=UnstructuredPdfPartitioner())
    # these are here mostly as examples; the last materialize will at this point take
    # effect, you can use these during testing.
    .materialize(path="/tmp/after_partition", source_mode=sycamore.MaterializeSourceMode.IF_PRESENT)
    .map(add_property_to_schema)
    .extract_properties(property_extractor=OpenAIPropertyExtractor(llm=llm, num_of_elements=35))
    .materialize(path="/tmp/after_llm", source_mode=sycamore.MaterializeSourceMode.IF_PRESENT)
    .merge(GreedyTextElementMerger(tokenizer, 300))
    .map(convert_timestamp)
    .spread_properties(["entity", "path"])
    .explode()
    .sketch()
    .embed(embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2"))
    .materialize(path="s3://aryn-public/materialize/examples/luna/ntsb_loader", source_mode=MaterializeSourceMode.IF_PRESENT)
    # materialize locally after reading from S3, it's a fair bit faster if you're running rmeotely
    .materialize(path="/tmp/after_embed", source_mode=sycamore.MaterializeSourceMode.IF_PRESENT)
    .write.opensearch(
        os_client_args=os_client_args,
        index_name=INDEX,
        index_settings=index_settings,
    )
)
