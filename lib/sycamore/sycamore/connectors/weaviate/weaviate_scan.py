from dataclasses import asdict
from sycamore.connectors.weaviate.weaviate_writer import WeaviateClientParams
from sycamore.plan_nodes import Scan
from sycamore.data import Document
from weaviate import WeaviateClient
from ray.data import Dataset, from_items
from sycamore.connectors.common import unflatten_data


class WeaviateScan(Scan):
    def __init__(self, collection_name: str, connection_params: WeaviateClientParams, **kwargs):
        super().__init__(**kwargs)
        self._collection_name = collection_name
        self._connection_params = connection_params

    def execute(self, **kwargs) -> Dataset:
        documents = []
        with WeaviateClient(**asdict(self._connection_params)) as wcl:
            collection = wcl.collections.get(self._collection_name)
            for object in collection.iterator(include_vector=True):
                doc = Document(
                    object.vector | unflatten_data(object.properties, "--") | {"doc_id": str(object.uuid)}
                )  # type: ignore
                documents.append(doc)

        return from_items(items=[{"doc": doc.serialize()} for doc in documents])

    def format(self):
        return "weaviate"
