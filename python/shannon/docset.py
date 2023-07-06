import logging

from shannon.data import Document
from shannon.execution import Node
from typing import (Callable, List, Optional)

from shannon import Context
from shannon.writer import DocSetWriter

logger = logging.getLogger(__name__)


class DocSet:
    """DocFrame is a distributed computation framework for Documents."""

    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan

    def lineage(self) -> Node:
        return self.plan

    def explain(self) -> None:
        # TODO, print out nice format DAG
        pass

    def show(self, limit: int = 20) -> None:
        dataset = self.context.execution.execute(self.plan)
        for row in dataset.take(limit):
            print(row)

    def partition_pdf(
            self,
            col_name: str,
            *,
            include_page_breaks: bool = False,
            strategy: str = "auto",
            infer_table_structure: bool = False,
            ocr_languages: str = "eng",
            max_partition: Optional[int] = 1500,
            include_metadata: bool = True,
            **resource_args) -> "DocSet":
        from shannon.execution.transforms.partition import PartitionPDF
        plan = PartitionPDF(
            self.plan, col_name,
            include_page_breaks=include_page_breaks,
            strategy=strategy,
            infer_table_structure=infer_table_structure,
            ocr_languages=ocr_languages,
            max_partition=max_partition,
            include_metadata=include_metadata, **resource_args)
        return DocSet(self.context, plan)

    def sentence_transformer_embed(
            self,
            col_name: str,
            *,
            model_name: str,
            embed_name: str = None,
            batch_size: int = None,
            device: str = None,
            **resource_args) -> "DocSet":
        from shannon.execution.transforms import SentenceTransformerEmbedding
        embedding = SentenceTransformerEmbedding(
            self.plan,
            col_name=col_name,
            model_name=model_name,
            embed_name=embed_name,
            batch_size=batch_size,
            device=device,
            **resource_args)
        return DocSet(self.context, embedding)

    def map(self, f: Callable[[Document], Document]) -> "DocSet":
        from shannon.execution.transforms.mapping import Map
        mapping = Map(self.plan, f=f)
        return DocSet(self.context, mapping)

    def flat_map(
            self,
            f: Callable[[Document], List[Document]],
            **kwargs) -> "DocSet":
        from shannon.execution.transforms.mapping import FlatMap
        flat_map = FlatMap(self.plan, f=f, **kwargs)
        return DocSet(self.context, flat_map)

    def map_batch(
            self,
            f: Callable[[List[Document]], List[Document]],
            **kwargs) -> "DocSet":
        from shannon.execution.transforms.mapping import MapBatch
        map_batch = MapBatch(self.plan, f=f, **kwargs)
        return DocSet(self.context, map_batch)

    @property
    def write(self) -> DocSetWriter:
        return DocSetWriter(self.context, self.plan)
