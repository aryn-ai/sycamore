from sycamore.plan_nodes import Scan
from sycamore.data import Document
from pinecone import PineconeApiException
from pinecone.grpc import PineconeGRPC
from ray.data import Dataset, from_items


class PineconeScan(Scan):
    def __init__(self, index_name: str, api_key: str, namespace: str = "", **kwargs):
        super().__init__(**kwargs)
        self._index_name = index_name
        self._api_key = api_key
        self._namespace = namespace

    def execute(self) -> Dataset:
        documents = []
        try:
            client = PineconeGRPC(api_key=self._api_key)
            index = client.Index(self._index_name)
            ids = []
            for pids in index.list(namespace=self._namespace):
                ids.extend(pids)
            for id, data in dict(index.fetch(ids=ids, namespace=self._namespace)["vectors"]).items():
                doc = Document({"doc_id": id, "embedding": data.values} | data.metadata)  # type: ignore
                documents.append(doc)
        except PineconeApiException as e:
            print(f"Read Request Failed: {e}")
        return from_items(items=[{"doc": doc.serialize()} for doc in documents])

    def format(self):
        return "pinecone"
