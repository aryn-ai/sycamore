from typing import Dict
import math

import numpy as np
import ray
from ray.data import (ActorPoolStrategy, Dataset)
from sentence_transformers import SentenceTransformer

from sycamore.execution import (Node, Transform)


class SentenceTransformerEmbedding(Transform):
    """Embedding based on HuggingFace Sentence Transformer"""
    def __init__(
            self,
            child: Node,
            model_name: str,
            batch_size: int = None,
            device: str = None,
            **resource_args):
        super().__init__(child, **resource_args)
        self.type = type
        self.model_name = model_name
        if device is None:
            import torch.cuda
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.batch_size = batch_size

    class SentenceTransformer:
        def __init__(
                self, model_name: str, batch_size: int = 100,
                device: str = None):
            self._transformer = SentenceTransformer(model_name)
            self._batch_size = batch_size
            self._device = device

        # TODO, embedding should consider entity type
        def __call__(
                self, doc_batch: Dict[str, np.ndarray]) -> \
                Dict[str, np.ndarray]:
            text_batch = doc_batch["content"].tolist()
            text_list = [text["text"] for text in text_batch]
            embeddings = self._transformer.encode(
                text_list, batch_size=self._batch_size, device=self._device)
            doc_batch.update({"embedding": embeddings.tolist()})

            return doc_batch

    def execute(self) -> "Dataset":
        dataset = self.child().execute()

        if self.device == "cuda":
            gpus = ray.available_resources().get("GPU")
            if "num_gpus" not in self.resource_args or \
                    self.resource_args["num_gpus"] <= 0:
                raise RuntimeError("Invalid GPU Nums!")
            gpu_per_task = self.resource_args["num_gpus"]

            output = dataset.map_batches(
                SentenceTransformerEmbedding.SentenceTransformer,
                batch_size=self.batch_size,
                compute=ActorPoolStrategy(
                    min_size=1, max_size=math.ceil(gpus/gpu_per_task)),
                fn_constructor_kwargs={
                    'model_name': self.model_name,
                    'batch_size': self.batch_size,
                    'device': self.device},
                num_gpus=gpu_per_task,
                **self.resource_args)
        else:
            # in case of no gpu required, we use tasks to make it easier
            # to be fusible
            embedder = self.SentenceTransformer(
                self.model_name, self.batch_size)
            output = dataset.map_batches(
                embedder.__call__,
                batch_size=self.batch_size,
                **self.resource_args)

        return output
