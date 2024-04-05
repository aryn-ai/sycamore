import docker
from docker.models.containers import Container
from opensearchpy import OpenSearch
from integration.containers.running import docker_compose
from integration.ingests.index import IndexInfo
import time

PROFILE_TO_NAME_MAP = {"sort-one": "sycamore_crawler_http_sort_one", "sort-all": "sycamore_crawler_http_sort_all"}
DEFAULT_INDEX_NAME = "demoindex0"


class HttpCrawlerIndex:
    def __init__(
        self,
        profile: str,
        opensearch: OpenSearch,
        importer: Container,
    ):
        self._profile = profile
        self._opensearch = opensearch
        self._importer = importer

    def __enter__(self):
        docker_client = docker.from_env()
        service_name = self._get_service_name()
        compose = docker_compose(services=[service_name])
        files = set()
        start_crawler_time = time.time()
        compose.start()
        crawler_container = compose.get_container(service_name=service_name, include_all=True)
        crawler_container = docker_client.containers.get(crawler_container.ID)
        assert crawler_container.wait().get("StatusCode") == 0, "Crawler container failed"
        logs = [log.decode() for log in crawler_container.logs().splitlines()]
        for log in reversed(logs):
            if "Spider opened" in log:
                break
            if log.startswith("Store"):
                pieces = log[6:].split(" as ")
                file = pieces[1].strip()
                if "unknown" not in file:
                    files.add(file)
        num_files = len(files)
        importer_logs = self._importer.logs(stream=True, since=start_crawler_time)
        for log in importer_logs:
            log = log.decode()
            if log.startswith("Successfully imported:"):
                list_insides = log.rstrip()[len("Successfully imported: [") : -len("]")]
                quoted_files = list_insides.split(",")
                imported_files = [file.strip()[len("'/app/") : -len("'")] for file in quoted_files]
                for file in imported_files:
                    print(f"imported {file}")
                    files.remove(file)
                print(f"remaining files: {files}")
            if len(files) == 0:
                print("finished importing!")
                break
            if log.startswith("No changes") and len(files) != num_files:
                raise RuntimeError(
                    "Importer thinks there are no more files to ingest, but there are more files to ingest"
                )
        return IndexInfo(name=DEFAULT_INDEX_NAME, num_docs=num_files)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._opensearch.indices.delete(index=DEFAULT_INDEX_NAME)

    def _get_service_name(self):
        return PROFILE_TO_NAME_MAP[self._profile]
