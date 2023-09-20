from ray.data import Dataset

from sycamore.data import Document
from sycamore.execution import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.execution.transforms.mapping import generate_flat_map_function


class Explode(SingleThreadUser, NonGPUUser, Transform):
    def __init__(self, child: Node, **resource_args):
        super().__init__(child, **resource_args)

    class ExplodeCallable:
        @staticmethod
        def explode(parent: Document) -> list[Document]:
            documents: list[Document] = [parent]

            import uuid

            for element in parent.elements:
                cur = Document(element.to_dict())
                cur.doc_id = str(uuid.uuid1())
                cur.parent_id = parent.doc_id
                documents.append(cur)
            del parent.elements
            return documents

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        exploder = Explode.ExplodeCallable()
        return dataset.flat_map(generate_flat_map_function(exploder.explode))
