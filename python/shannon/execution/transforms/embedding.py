from ray.data import (ActorPoolStrategy, Dataset)
from shannon.execution import (Node, Transform)
from shannon.execution.kernels import SentenceTransformerEmbeddingKernel


class Embedding(Transform):
    def __init__(
            self,
            child: Node,
            col_name: str,
            model_name: str,
            embed_name: str = None,
            batch_size: int = None,
            **resource_args):
        super().__init__(child, **resource_args)
        self.col_name = col_name
        self.model_name = model_name
        self.embed_name = embed_name
        self.batch_size = batch_size

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def get_batch_size(self) -> int:
        return self.batch_size


class SentenceTransformerEmbedding(Embedding):
    def __init__(
            self,
            child: Node,
            col_name: str,
            model_name: str,
            embed_name: str = None,
            batch_size: int = None,
            device: str = None,
            **resource_args):
        super().__init__(
            child, col_name=col_name, model_name=model_name,
            embed_name=embed_name, batch_size=batch_size, **resource_args)
        self.device = device
        self.batch_size = batch_size

    def execute(self) -> "Dataset":
        dataset = self.child().execute()
        block_count = dataset.num_blocks()
        output = dataset.map_batches(
            SentenceTransformerEmbeddingKernel,
            batch_size=self.batch_size,
            compute=ActorPoolStrategy(min_size=1, max_size=block_count),
            fn_constructor_kwargs={
                'col_name': self.col_name,
                'model_name': self.model_name,
                'batch_size': self.batch_size,
                'device': self.device},
            **self.resource_args)
        return output
