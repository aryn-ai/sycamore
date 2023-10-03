import math
from abc import ABC, abstractmethod
from typing import Optional

import ray
from ray.data import ActorPoolStrategy, Dataset
from sentence_transformers import SentenceTransformer

from sycamore.data import Document
from sycamore.plan_nodes import Node, Transform
from sycamore.transforms.map import generate_map_batch_function, generate_map_batch_class_from_callable


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
    """
    SentenceTransformerEmbedder is an Embedder class for generating sentence embeddings using the
    SentenceTransformer model.

    Args:
        model_name: The name or path of the SentenceTransformer model to use for embedding.
        batch_size: The dataset batch size for embedding, if specified. Default is None.
        model_batch_size: The batch size used by the underlying SentenceTransformer model for embedding.
        device: The device (e.g., "cpu" or "cuda") on which to perform embedding.

    Example:
        .. testcode::

            model_name = "bert-base-nli-mean-tokens"
            embedder = SentenceTransformerEmbedder(model_name)
            embedded_documents = embedder.generate_embeddings(document_batch)

    """

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
    """
    Embed is a transformation that generates embeddings a docset using an Embedder.

    The generated embeddings are stored in a special embedding property on each document.
    It utilizes an Embedder to perform the embedding process.

    Args:
        child: The source node or component that provides the dataset to be embedded.
        embedder: An instance of an Embedder class that defines the embedding method to be applied.
        resource_args: Additional resource-related arguments that can be passed to the embedding operation.

    Example:
        .. testcode::

            source_node = ...  # Define a source node or component that provides a dataset.
            custom_embedder = MyEmbedder(embedding_params)
            embed_transform = Embed(child=source_node, embedder=custom_embedder)
            embedded_dataset = embed_transform.execute()
    """

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
