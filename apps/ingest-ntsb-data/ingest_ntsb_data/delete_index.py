#!/usr/bin/env python

# This script can be used to delete an index from a local OpenSearch instance.

import argparse

from opensearchpy import OpenSearch


def main():
    parser = argparse.ArgumentParser(description="Delete an index from OpenSearch.")
    parser.add_argument("index_name", type=str, help="The name of the index to delete.")
    args = parser.parse_args()

    # create an OpenSearch client
    os_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }
    client = OpenSearch(**os_client_args)

    # check if the index exists
    if client.indices.exists(index=args.index_name):
        # delete the index
        response = client.indices.delete(index=args.index_name)
        print(f"Deleted index: {args.index_name}")
        print(response)
    else:
        print(f"Index '{args.index_name}' does not exist.")


if __name__ == "__main__":
    main()
