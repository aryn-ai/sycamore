import json
import time

import docker
from pathlib import Path

import pytest
import requests
from opensearchpy import OpenSearch


@pytest.fixture(scope="session")
def opensearch_client():
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True,
        http_auth=("admin", "admin"),
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=120
    )


@pytest.fixture(scope="module")
def setup_containers(opensearch_client):
    opensearch_image = build_plugin_container()
    server_image = build_service_container()
    network, opensearch_container, rps_container = run_compose(server_image, opensearch_image)
    wait_for_opensearch_connection(opensearch_client)
    yield opensearch_container, rps_container
    opensearch_container.stop()
    rps_container.stop()
    network.remove()


def build_plugin_container():
    image, logs = docker.DockerClient().images.build(path=".", tag="test-rps-os", dockerfile="docker/Dockerfile")
    return image


def build_service_container():
    image, logs = docker.DockerClient().images.build(path=".", tag="test-rps")
    return image


def run_compose(server_image, opensearch_image):
    client = docker.DockerClient()
    nets = client.networks.list(names=["test-net"])
    if len(nets) == 0:
        network = client.networks.create("test-net")
    else:
        network = nets[0]
    opensearch_container = client.containers.run(
        image=opensearch_image,
        name="test-opensearch",
        network=network.id,
        detach=True,
        ports={"9200/tcp": 9200, "9600/tcp": 9600, "9300/tcp": 9300},
        environment={"discovery.type": "single-node"},
        remove=True
    )
    rps_container = client.containers.run(
        image=server_image,
        name="test-rps",
        network=network.id,
        detach=True,
        ports={"2796/tcp": 2796},
        remove=True
    )
    return network, opensearch_container, rps_container


@pytest.fixture(scope="module", params=[Path("test/resources/sb_processed.jsonl")])
def upload_jsonl_index(request, setup_containers, opensearch_client):
    path = request.param
    index_name = Path(path).stem
    opensearch_client.indices.create(index=index_name, body={
        "settings": {"index.knn": True, "number_of_shards": 5, "number_of_replicas": 1},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {"name": "hnsw", "engine": "nmslib"},
                }
            }
        }
    })
    with open(path, "r") as f:
        records = []
        for line in f:
            records.append(json.loads(line))
            if len(records) == 100:
                bulk_actions = []
                for r in records:
                    bulk_actions.append({"index": {"_index": index_name, "_id": r["_id"]}})
                    bulk_actions.append(r["_source"])
                opensearch_client.bulk(body=bulk_actions)
                records = []
        if len(records) > 0:
            bulk_actions = []
            for r in records:
                bulk_actions.append({"index": {"_index": index_name, "_id": r["_id"]}})
                bulk_actions.append(r["_source"])
            opensearch_client.bulk(body=bulk_actions)
    yield index_name
    opensearch_client.indices.delete(index=index_name)


def wait_for_opensearch_connection(client: OpenSearch):
    while not client.ping():
        print("ping")
        time.sleep(1)


@pytest.fixture
def singleton_pipeline(request, setup_containers):
    processor_name = request.node.get_closest_marker("processor_name")
    if processor_name is None:
        raise ValueError("Missing \"processor_name\" mark")
    processor_name = processor_name.args[0]
    pipeline_name = processor_name + "-pipeline"
    pipeline_def = {
        "response_processors": [
            {
                "remote_processor": {
                    "endpoint": "test-rps:2796/RemoteProcessorService/ProcessResponse",
                    "processor_name": processor_name
                }
            }
        ]
    }
    requests.put(
        url=f"http://localhost:9200/_search/pipeline/{pipeline_name}",
        json=pipeline_def
    )
    yield pipeline_name
    requests.delete(
        url=f"http://localhost:9200/_search/pipeline/{pipeline_name}"
    )

