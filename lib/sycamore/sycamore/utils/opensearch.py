import os


def guess_opensearch_host():
    # The OpenSearch instance to use.
    if os.path.exists("/.dockerenv"):
        opensearch_host = "opensearch"
        print("Assuming we are in a Sycamore Jupyter container, using opensearch for OpenSearch host")
    else:
        opensearch_host = "localhost"
        print("Assuming we are running outside of a container, using localhost for OpenSearch host")
    return opensearch_host
