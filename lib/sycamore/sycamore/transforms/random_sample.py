import random
from typing import Optional, TYPE_CHECKING

from sycamore.plan_nodes import Node, Transform
from sycamore.data import Document

if TYPE_CHECKING:
    from ray.data import Dataset


class RandomSample(Transform):
    """
    Generates a random sample of documents in a collection.

    Args:
        child: The plan node providing the dataset.
        fraction: The fraction of documents to retain.
        seed: The seed to use to initialize the RNG.
        resource_args: Additional resource-related arguments to pass to the execution env.
    """

    def __init__(self, child: Node, fraction: float, seed: Optional[int] = None, **resource_args):
        super().__init__(child, **resource_args)
        self.fraction = fraction
        self.seed = seed

    def execute(self, **kwargs) -> "Dataset":
        dataset = self.child().execute()
        return dataset.random_sample(self.fraction, seed=self.seed)

    def local_execute(self, all_docs: list[Document]) -> list[Document]:
        if self.seed is not None:
            random.seed(self.seed)
        return random.sample(all_docs, int(len(all_docs) * self.fraction))
