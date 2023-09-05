from typing import Dict

import numpy as np
from ray.data import (ActorPoolStrategy, Dataset)
from sentence_transformers import SentenceTransformer

from sycamore.execution import (Node, Transform)


class SentenceTransformerEmbedding(Transform):
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
            text_batch = doc_batch["text_representation"].tolist()
            embeddings = self._transformer.encode(
                text_batch, batch_size=self._batch_size, device=self._device)
            doc_batch.update({"embedding": embeddings.tolist()})

            return doc_batch

    def execute(self) -> "Dataset":
        dataset = self.child().execute()
        block_count = dataset.num_blocks()
        output = dataset.map_batches(
            SentenceTransformerEmbedding.SentenceTransformer,
            batch_size=self.batch_size,
            compute=ActorPoolStrategy(min_size=1, max_size=block_count),
            fn_constructor_kwargs={
                'model_name': self.model_name,
                'batch_size': self.batch_size,
                'device': self.device},
            **self.resource_args)
        return output
