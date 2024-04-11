import pytest
import docker
import requests
import time
from opensearchpy import OpenSearch
from testcontainers.compose.compose import DockerCompose
from typing import List
from integration import SYCAMORE_ROOT


def docker_compose(services: List[str] = []):
    """
    Get a docker compose object that controls some set of the sycamore stack
    :param services: the list of services to control. [] -> all services
    :return: the docker compose object
    """
    if len(services) > 0:
        return DockerCompose(SYCAMORE_ROOT, services=services)
    else:
        return DockerCompose(SYCAMORE_ROOT)


@pytest.fixture(scope="session")
def container_urls():
    """
    Get a mapping from container/service to network address
    :return: map of container/service to network address
    """
    return {
        "opensearch": ("localhost", 9200),
        "rps": ("localhost", 2796),
        "demo-ui": ("localhost", 3000),
        "jupyter": ("localhost", 8888),
    }


@pytest.fixture(scope="session")
def container_handles():
    """
    Get docker objects representing each container in the stack
    :return: mapping from container/service to docker objects
    """
    docker_client = docker.from_env()
    jupyter = docker_client.containers.list(filters={"label": "com.docker.compose.service=jupyter"})[0]
    opensearch = docker_client.containers.list(filters={"label": "com.docker.compose.service=opensearch"})[0]
    demo_ui = docker_client.containers.list(filters={"label": "com.docker.compose.service=demo-ui"})[0]
    rps = docker_client.containers.list(filters={"label": "com.docker.compose.service=rps"})[0]
    importer = docker_client.containers.list(filters={"label": "com.docker.compose.service=importer"})[0]
    return {"jupyter": jupyter, "opensearch": opensearch, "demo_ui": demo_ui, "rps": rps, "importer": importer}


@pytest.fixture(scope="session")
def opensearch_client(container_urls):
    """
    Get an opensearch client that hits the stack opensearch
    :return: the opensearch client
    """
    host, port = container_urls["opensearch"]
    urlstr = f"https://{host}:{port}"
    # Ten minute deadline for opensearch startup
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            r = requests.get(f"{urlstr}/_cluster/settings", verify=False)
            if r.status_code == 200 and "aryn_deploy_complete" in r.text:
                print("Ready")
                break
        except requests.exceptions.ConnectionError:
            pass

        time.sleep(1)

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,  # enables gzip compression for request bodies
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
