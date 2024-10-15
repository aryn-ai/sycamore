#!/usr/bin/env python

# This script will populate a local OpenSearch index, using Sycamore, with data
# from NTSB incident reports. This is useful for testing the Sycamore Query UI
# with a real dataset.
#
# Run with: poetry run python queryui/loader.py [--delete]

import argparse
import json
import logging
import os
import tempfile

import sycamore
from sycamore.data import Document
from sycamore.transforms.partition import ArynPartitioner
from sycamore.functions import HuggingFaceTokenizer
from sycamore.transforms import AssignDocProperties, DateTimeStandardizer, USStateStandardizer
from sycamore.transforms.extract_schema import (
    OpenAIPropertyExtractor,
)
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms.embed import SentenceTransformerEmbedder
from opensearchpy import OpenSearch

logging.basicConfig(level=logging.INFO)


def add_schema_property(doc: Document) -> Document:
    """Add a _schema and _schema_class property to the document with the NTSB data schema."""
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
            "operator": {"type": "string"},
            "conditions": {"type": "string"},
            "lowestCloudCondition": {"type": "string"},
            "lowestCeiling": {"type": "string"},
            "conditionOfLight": {"type": "string"},
            "temperature": {"type": "string"},
            "windSpeed": {"type": "string"},
            "windDirection": {"type": "string"},
            "visibility": {"type": "string"},
            "departureAirport": {"type": "string"},
            "destinationAirport": {"type": "string"},
        },
    }
    doc.properties.update({"_schema": schema_json, "_schema_class": "Flight Accident Report"})
    return doc


def standardize_location(doc: Document, key_path: list[str]) -> Document:
    try:
        doc = USStateStandardizer.standardize(doc, key_path=key_path)
    except KeyError:
        logging.warning(f"Key {key_path} not found in document: {doc}")
    return doc


def main():
    argparser = argparse.ArgumentParser(prog="loaddata")
    argparser.add_argument("--dump", action="store_true", help="Dump contents of existing index and exit")
    argparser.add_argument("--delete", action="store_true", help="Delete the index if it already exists")
    argparser.add_argument(
        "--tempdir", default=tempfile.gettempdir(), help="Temporary directory for materialize output"
    )
    argparser.add_argument("--oshost", default=None, help="OpenSearch host")
    argparser.add_argument("--osport", default=9200, help="OpenSearch port")
    argparser.add_argument("--index", default="const_ntsb", help="OpenSearch index name")
    argparser.add_argument("--source", default="s3://aryn-public/ntsb/", help="Source data path")
    argparser.add_argument("--limit", default=None, type=int, help="Limit the number of documents to process")
    args = argparser.parse_args()

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

    if args.dump:
        # This will dump up to the first 10000 documents in the index.
        # TODO: Implement pagination to handle larger indices.
        contents = os_client.search(index=args.index, body={"query": {"match_all": {}}}, size=args.limit or 10000)
        hits = contents["hits"]["hits"]
        for hit in hits:
            print(json.dumps(hit, indent=2))
        return

    if os_client.indices.exists(index=args.index):
        if args.delete:
            print(f"Index {args.index} already exists, deleting as requested")
            os_client.indices.delete(index=args.index)
        else:
            raise Exception(f"Index {args.index} already exists. Run with --delete to delete it.")

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
    docset = context.read.binary(args.source, binary_format="pdf")
    if args.limit:
        docset = docset.limit(args.limit)
    tokenizer = HuggingFaceTokenizer("thenlper/gte-small")
    llm = OpenAI(OpenAIModels.GPT_4O.value)

    partitioned_docset = (
        docset.partition(partitioner=ArynPartitioner(extract_table_structure=True, use_ocr=True, extract_images=True))
        .materialize(path=f"{args.tempdir}/ntsb-loader-stage-0", source_mode=sycamore.MATERIALIZE_USE_STORED)
        .map(add_schema_property)
        .materialize(path=f"{args.tempdir}/ntsb-loader-stage-1", source_mode=sycamore.MATERIALIZE_USE_STORED)
        .map(
            lambda doc: AssignDocProperties.assign_doc_properties(
                doc, element_type="table", property_name="table_props"
            )
        )
        .extract_properties(property_extractor=OpenAIPropertyExtractor(llm=llm, num_of_elements=35))
        .merge(GreedyTextElementMerger(tokenizer, 300))
        .map(lambda doc: standardize_location(doc, key_path=["properties", "entity", "location"]))
        .map(lambda doc: standardize_location(doc, key_path=["properties", "entity", "departureAirport"]))
        .map(lambda doc: standardize_location(doc, key_path=["properties", "entity", "destinationAirport"]))
        .map(lambda doc: DateTimeStandardizer.standardize(doc, key_path=["properties", "entity", "dateAndTime"]))
        .spread_properties(["entity", "path"])
        .explode()
        .sketch()
        .embed(
            embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
    )

    for doc in partitioned_docset.take_all():
        print(str(doc))

    partitioned_docset.write.opensearch(
        os_client_args=os_client_args,
        index_name=args.index,
        index_settings=index_settings,
    )


if __name__ == "__main__":
    main()
