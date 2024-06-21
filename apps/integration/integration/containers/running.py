import pytest
import docker
import requests
import time
import logging
import subprocess
from opensearchpy import OpenSearch


IMPORTANT_PORTS = {
    "opensearch": 9200,
    "rps": 2796,
    "demo-ui": 3000,
    "jupyter": 8888,
}


@pytest.fixture(scope="session")
def container_urls(stack):
    """
    Get a mapping from container/service to network address
    :return: map of container/service to network address
    """
    ret = {
        service_name: stack.get_service_host_and_port(service_name, port=IMPORTANT_PORTS[service_name])
        for service_name in [c.Service for c in stack.get_containers()]
        if service_name in IMPORTANT_PORTS
    }
    logging.error(f"Container URLs {ret}")
    return ret


@pytest.fixture(scope="session")
def container_handles(stack):
    """
    Get docker objects representing each container in the stack
    :return: mapping from container/service to docker objects
    """
    logging.error("Getting clients...")
    subprocess.run(["docker", "ps"])
    docker_client = docker.from_env()
    jupyter = docker_client.containers.get(stack.get_container("jupyter").ID)
    opensearch = docker_client.containers.get(stack.get_container("opensearch").ID)
    demo_ui = docker_client.containers.get(stack.get_container("demo-ui").ID)
    rps = docker_client.containers.get(stack.get_container("rps").ID)
    importer = docker_client.containers.get(stack.get_container("importer").ID)
    ret = {"jupyter": jupyter, "opensearch": opensearch, "demo_ui": demo_ui, "rps": rps, "importer": importer}
    logging.error(f"Got all clients: {ret}")
    return ret


@pytest.fixture(scope="session")
def opensearch_client(container_urls):
    """
    Get an opensearch client that hits the stack opensearch
    :return: the opensearch client
    """
    logging.error("Waiting for opensearch to start...")
    host, port = container_urls["opensearch"]
    urlstr = f"https://{host}:{port}"
    # Ten minute deadline for opensearch startup
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            r = requests.get(f"{urlstr}/_cluster/settings", verify=False)
            if r.status_code == 200 and "aryn_deploy_complete" in r.text:
                logging.error("Ready")
                break
        except requests.exceptions.ConnectionError:
            pass

        time.sleep(3)
        logging.error("Still waiting for opensearch to start...")

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,  # enables gzip compression for request bodies
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=20,
    )
