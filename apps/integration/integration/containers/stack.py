from testcontainers.compose import DockerCompose
import pytest
from integration import SYCAMORE_ROOT
from typing import List


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
def stack():
    base_compose = docker_compose()
    with base_compose:
        yield base_compose
