from testcontainers.compose import DockerCompose
import pytest
from integration import SYCAMORE_ROOT
from typing import List, Union
import logging
import subprocess
import os
import time


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
    logging.error(f"Setting up compose with tag {tag} .. this may take a long time")
    with base_compose:
        logging.error("Done")
        yield base_compose

        logs = open("docker-logs.txt", "w")
        subprocess.call(["docker", "compose", "logs"], stdout=logs)
        logging.error("Wrote logs to docker-logs.txt")
        maybe_wait_to_abort()
        logging.error("Shutting down docker, this can take a little bit")
    set_version_tag("stable")


def maybe_wait_to_abort():
    if os.environ.get("NOEXIT", "") == "":
        logging.error("You can set the environment variable NOEXIT to force a pause at this point")
        return

    while True:
        logging.error("Waiting for /tmp/abort to exist")
        if os.path.isfile("/tmp/abort"):
            break
        time.sleep(1)
