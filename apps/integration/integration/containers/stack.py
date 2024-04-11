from testcontainers.compose import DockerCompose
import pytest
from integration import SYCAMORE_ROOT
from typing import List, Union


def docker_compose(services: Union[List[str], None] = None):
    """
    Get a docker compose object that controls some set of the sycamore stack
    :param services: the list of services to control. [] -> all services
    :return: the docker compose object
    """
    if services:
        return DockerCompose(SYCAMORE_ROOT, services=services)
    else:
        return DockerCompose(SYCAMORE_ROOT)


def set_version_tag(tag: str):
    with open(SYCAMORE_ROOT / ".env", "r") as f:
        lines = f.readlines()
    version_pos = [line[: len("VERSION=")] for line in lines].index("VERSION=")
    lines[version_pos] = f"VERSION={tag}\n"
    with open(SYCAMORE_ROOT / ".env", "w") as f:
        f.writelines(lines)


@pytest.fixture(scope="session")
def tag(request):
    return request.config.getoption("--docker-tag")


@pytest.fixture(scope="session")
def stack(tag):
    base_compose = docker_compose()
    base_compose.pull = True
    set_version_tag(tag)
    with base_compose:
        yield base_compose
    set_version_tag("stable")
