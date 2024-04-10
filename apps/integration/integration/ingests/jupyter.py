from docker.models.containers import Container
from opensearchpy import OpenSearch

from integration.ingests.index_info import IndexInfo


JUPYTER_NB_INFO = {
    "default-prep-script.ipynb": IndexInfo(name="demoindex0", num_docs=2),
    "jupyter_dev_example.ipynb": IndexInfo(name="local_development_example_index_withentity", num_docs=2),
}


class JupyterIndex:
    def __init__(self, nb_name: str, opensearch: OpenSearch, jupyter: Container):
        self._nb_name = nb_name
        self._opensearch = opensearch
        self._jupyter = jupyter
        if nb_name not in JUPYTER_NB_INFO:
            raise ValueError(f"Unrecognized notebook name: {nb_name}")
        self._index_info = JUPYTER_NB_INFO[nb_name]

    def __enter__(self):
        command = [
            "poetry",
            "run",
            "jupyter",
            "nbconvert",
            "--execute",
            "--stdout",
            "--debug",
            "--to",
            "markdown",
            f"work/examples/{self._nb_name}",
        ]
        exc, logs = self._jupyter.exec_run(command, stream=True)
        for log in logs:
            print(log.decode("utf-8").rstrip())
        return self._index_info

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._opensearch.indices.delete(index=self._index_info.name)
