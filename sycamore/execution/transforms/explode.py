from typing import Any

from ray.data import Dataset

from sycamore.data import Document
from sycamore.execution import Node, Transform, SingleThreadUser, NonGPUUser


class Explode(SingleThreadUser, NonGPUUser, Transform):
    def __init__(self, child: Node, **resource_args):
        super().__init__(child, **resource_args)

    class ExplodeCallable:
        @staticmethod
        def explode(dict: dict[str, Any]) -> list[dict[str, Any]]:
            parent = Document(dict)
            documents = [parent]

            import uuid

            for element in parent.elements:
                cur = Document(element.to_dict())
                cur.doc_id = str(uuid.uuid1())
                cur.parent_id = parent.doc_id
                documents.append(cur)
            del parent.elements
            return [document.to_dict() for document in documents]

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        exploder = Explode.ExplodeCallable()
        return dataset.flat_map(exploder.explode)
