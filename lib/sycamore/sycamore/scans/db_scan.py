from opensearchpy import OpenSearch
from ray.data import Dataset, from_items

from sycamore.data import Document
from sycamore.plan_nodes import Scan


class OpenSearchScan(Scan):
    def __init__(
        self,
        index_name: str,
        os_client_args: dict,
        query: dict = None,
        **resource_args,
    ):
        super().__init__(**resource_args)
        self.index_name = index_name
        self.os_client_args = os_client_args
        self.os_client = OpenSearch(**os_client_args)
        self.query = query

    def execute(self) -> Dataset:
        result = []

        scroll = "1m"
        response = self.os_client.search(index=self.index_name, scroll=scroll, size=200, body=self.query)
        scroll_id = response["_scroll_id"]

        while True:
            hits = response["hits"]["hits"]
            for hit in hits:
                result += [Document(hit["_source"])]

            if not hits:
                break
            response = self.os_client.scroll(scroll_id=scroll_id, scroll=scroll)
        return from_items(items=[{"doc": doc.serialize()} for doc in result])

    def format(self):
        return "opensearch"
