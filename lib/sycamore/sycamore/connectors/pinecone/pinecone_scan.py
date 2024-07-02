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
                doc_id = id.str.split()[1] if len(id.str.split()) > 1 else id
                if data.sparse_vector:
                    term_frequency = dict(zip(data.sparse_vector.indices, data.sparse_vector.values))
                    data.metadata["term_frequency"] = term_frequency
                doc = Document({"doc_id": doc_id, "embedding": data.values} | data.metadata)  # type: ignore
                documents.append(doc)
        except PineconeApiException as e:
            print(f"Read Request Failed: {e}")
        return from_items(items=[{"doc": doc.serialize()} for doc in documents])

    def format(self):
        return "pinecone"
