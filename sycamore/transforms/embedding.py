import math
from abc import ABC, abstractmethod
from typing import Optional

import ray
from ray.data import ActorPoolStrategy, Dataset
from sentence_transformers import SentenceTransformer

from sycamore.data import Document
from sycamore.plan_nodes import Node, Transform
from sycamore.transforms.mapping import generate_map_batch_function, generate_map_batch_class_from_callable


class Embedder(ABC):
    def __init__(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
        model_batch_size: int = 100,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size

        if device is None:
            import torch.cuda

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def __call__(self, doc_batch: list[Document]) -> list[Document]:
        return self.generate_embeddings(doc_batch)

    @abstractmethod
    def generate_embeddings(self, doc_batch: list[Document]) -> list[Document]:
        pass


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        batch_size: Optional[int] = None,
        model_batch_size: int = 100,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, batch_size, model_batch_size, device)
        self.type = type
        self._transformer: Optional[SentenceTransformer] = None

    def generate_embeddings(self, doc_batch: list[Document]) -> list[Document]:
        if not self._transformer:
            self._transformer = SentenceTransformer(self.model_name)

        assert self._transformer is not None

        text_batch = [doc.text_representation for doc in doc_batch]
        embeddings = self._transformer.encode(text_batch, batch_size=self.model_batch_size, device=self.device)
        for doc, embedding in zip(doc_batch, embeddings):
            doc.embedding = embedding

        return doc_batch


class Embed(Transform):
    def __init__(self, child: Node, embedder: Embedder, **resource_args):
        super().__init__(child, **resource_args)
        self._embedder = embedder

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        if self._embedder.device == "cuda":
            available_gpus = ray.available_resources().get("GPU")
            if "num_gpus" not in self.resource_args:
                self.resource_args["num_gpus"] = 1
            if self.resource_args["num_gpus"] <= 0:
                raise RuntimeError("Invalid GPU Nums!")
            gpu_per_task = self.resource_args["num_gpus"]

            output = dataset.map_batches(
                generate_map_batch_class_from_callable(self._embedder.generate_embeddings),
                batch_size=self._embedder.batch_size,
                compute=ActorPoolStrategy(min_size=1, max_size=math.ceil(available_gpus / gpu_per_task)),
                **self.resource_args
            )
        else:
            # in case of no gpu required, we use tasks to make it easier
            # to be fusible
            output = dataset.map_batches(
                generate_map_batch_function(self._embedder.generate_embeddings),
                batch_size=self._embedder.batch_size,
                **self.resource_args
            )

        return output
