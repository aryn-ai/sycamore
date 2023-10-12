from ray.data import Dataset

from sycamore.data import Document
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_flat_map_function


class Explode(SingleThreadUser, NonGPUUser, Transform):
    """
    The Explode transform converts the elements of each document into top-level documents. For example, if you explode a
    DocSet with a single document containing two elements, the resulting DocSet will have three documents – the original
    plus a new Document for each of the elements.

    Args:
        child: The source node or component that provides the hierarchical documents to be exploded.
        resource_args: Additional resource-related arguments that can be passed to the explosion operation.

    Example:
        .. code-block:: python

            source_node = ...  # Define a source node or component that provides hierarchical documents.
            explode_transform = Explode(child=source_node)
            exploded_dataset = explode_transform.execute()
    """

    def __init__(self, child: Node, **resource_args):
        super().__init__(child, **resource_args)

    class ExplodeCallable:
        @staticmethod
        def explode(parent: Document) -> list[Document]:
            documents: list[Document] = [parent]

            import uuid

            for element in parent.elements:
                cur = Document(element.data)
                cur.doc_id = str(uuid.uuid1())
                cur.parent_id = parent.doc_id
                documents.append(cur)
            del parent.elements
            return documents

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        exploder = Explode.ExplodeCallable()
        return dataset.flat_map(generate_flat_map_function(exploder.explode))
