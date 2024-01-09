from ray import serve
import ray
from ray.serve.handle import DeploymentHandle
from sycamore.transforms.embed import Embedder

from typing import Any, Optional, Callable

from sentence_transformers import SentenceTransformer

from sycamore.data import Document






class RayServeSentenceTransformerEmbedder(Embedder):
    """_summary_

    Args:
        Embedder (_type_): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """

    @serve.deployment(num_replicas=1, )
    class RemoteSentenceTransformer:
        def __init__(self, model_name: str, model_batch_size: int = 100, device: Optional[str] = None):
            self._model_name = model_name
            self._model_batch_size = model_batch_size
            self._device = device
            self._transformer = SentenceTransformer(model_name_or_path=model_name, device=device)
        
        def embed(self, texts: list[str]):
            return self._transformer.encode(texts, batch_size=self._model_batch_size, device=self._device)

    def __init__(self,
        model_name: str,
        batch_size: Optional[int] = None,
        model_batch_size: int = 100,
        pre_process_document: Optional[Callable[[Document], str]] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model_name, batch_size, model_batch_size, pre_process_document, device)
        self._model_handle: DeploymentHandle = serve.run(RayServeSentenceTransformerEmbedder.RemoteSentenceTransformer.bind(model_name, model_batch_size, device))


    def generate_embeddings(self, doc_batch: list[Document]) -> list[Document]:
        
        text_batch = [self.pre_process_document(doc) for doc in doc_batch if doc.text_representation is not None]
        embeddings = ray.get(self._model_handle.embed.remote(text_batch))

        i = 0
        for doc in doc_batch:
            if doc.text_representation is not None:
                doc.embedding = embeddings[i].tolist()
                i += 1

        return doc_batch
