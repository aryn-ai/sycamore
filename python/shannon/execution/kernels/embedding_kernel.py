import numpy as np
from sentence_transformers import SentenceTransformer
from typing import (Dict, List)


class EmbeddingKernel:
    pass


class SentenceTransformerEmbeddingKernel(EmbeddingKernel):

    def __init__(
            self, col_name: str, model_name: str, embed_name: str = None,
            batch_size: int = 100, device: str = None):
        self._transformer = SentenceTransformer(model_name)
        self._col_name = col_name
        self._embed_name = embed_name if embed_name is not None \
            else "embedding_" + col_name
        self._batch_size = batch_size
        self._device = device

    def __call__(
            self, doc_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        text_batch: List[str] = doc_batch[self._col_name].tolist()
        embeddings = self._transformer.encode(
            text_batch, batch_size=self._batch_size, device=self._device)
        # Need to extract this out into
        doc_batch.update({self._embed_name: embeddings})
        return doc_batch
