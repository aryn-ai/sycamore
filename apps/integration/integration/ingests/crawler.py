import docker
from docker.models.containers import Container
from opensearchpy import OpenSearch
from integration.containers.stack import docker_compose
from integration.ingests.index_info import IndexInfo
from subprocess import CalledProcessError
import time
import logging
import subprocess
import opensearchpy.exceptions

PROFILE_TO_NAME_MAP = {"sort-one": "sycamore_crawler_http_sort_one", "sort-all": "sycamore_crawler_http_sort_all"}
DEFAULT_INDEX_NAME = "demoindex0"


class HttpCrawlerIndex:
    """
    Class that ingests an index using a sycamore http crawler preset
    """

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
        """
        Context manager for an ingest with sycamore http crawler
        Start a crawler image, wait for it to finish and read what it loads from the logs.
        Then watch the importer logs to determine when it has ingested all of the docs it needs to
        """
        logging.error("http_crawler:__enter__")
        docker_client = docker.from_env()
        logging.error("http_crawler:__enter__:rm")
        (code, out) = self._importer.exec_run(cmd="rm -rf /app/.scrapy/imported /app/.scrapy/downloads")
        logging.error(f"rm says {code} {out}")
        (code, out) = self._importer.exec_run(cmd="find /app/.scrapy ! -name httpcache -print")
        outstr = str(out)
        if code != 0 or "imported" in outstr or "downloads" in outstr:
            logging.error(f"Find says {code} {out}")
            assert False

        service_name = self._get_service_name()
        logging.error(f"cleanup container {service_name}")
        proc = subprocess.run(["docker", "compose", "rm", "-f", service_name])
        logging.error(f"{proc}")
        assert proc.returncode == 0
        logging.error("http_crawler:__enter__:pre-compose")
        compose = docker_compose(services=[service_name])
        files = set()
        start_crawler_time = time.time()
        try:
            compose.start()
        except CalledProcessError as e:
            logging.error(f"CPE {e}\n\n{e.output}\n\n{e.stderr}")
            raise
        logging.error("http_crawler:__enter__:started")
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
                    logging.info(f"found {file} in logs")

        if len(files) == 0:
            logging.error(f"Crawler {service_name} did not download any files?!")
            raise RuntimeError(f"Crawler {service_name} did not download any files")

        num_files = len(files)
        importer_logs = self._importer.logs(stream=True, since=start_crawler_time)
        no_change_count = 0
        for log in importer_logs:
            log = log.decode()
            if "Successfully imported:" in log or log.startswith("No changes"):
                # the log line and ray output can be intermingled. Inspect the filesystem to figure out what is imported
                (code, out) = self._importer.exec_run(cmd="find /app/.scrapy/imported -type f")
                if b"No such file or directory" in out:
                    # no files from before scraper finishes
                    continue
                foundfiles = out.decode().split("\n")

                logging.error(f"Expecting files: {files}")
                remain = set(files)
                for i in foundfiles:
                    if i == "":
                        continue
                    i = i.replace("/app/", "").replace("/imported/", "/downloads/")
                    if i not in remain:
                        logging.error(f"{i} not found in {remain}")
                        assert False
                    remain.remove(i)

                if len(remain) == 0:
                    logging.error("Found all files")
                    break

                # There is a race between no changes being noticed and the start of the log reporting.
                # Ignore a handful of them. If there is a bug where the importer stops early, this will
                # fail soon enough.
                logging.error(f"Remaining files: {remain}")
                if log.startswith("No changes"):
                    no_change_count = no_change_count + 1
                    if no_change_count >= 5:
                        raise RuntimeError("Importer stopped importing with remaining files")
                    else:
                        logging.error(f"No changes {no_change_count}/5 before aborting")
                else:
                    no_change_count = 0

        while not self._opensearch.indices.exists(DEFAULT_INDEX_NAME):
            logging.error(f"Waiting for index {DEFAULT_INDEX_NAME} to exist")
            time.sleep(1)

        return IndexInfo(name=DEFAULT_INDEX_NAME, num_docs=num_files)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit. Drop the index.
        """
        try:
            logging.error(f"Deleteing index {DEFAULT_INDEX_NAME}")
            self._opensearch.indices.delete(index=DEFAULT_INDEX_NAME)
        except opensearchpy.exceptions.NotFoundError as e:
            logging.error(f"Failed to delete not found index {e}")

    def _get_service_name(self):
        return PROFILE_TO_NAME_MAP[self._profile]
