import logging
import pprint
from typing import Callable, Optional, Any, Iterable

from sycamore import Context
from sycamore.data import Document
from sycamore.execution import Node
from sycamore.execution.transforms import Partition
from sycamore.execution.transforms.embedding import Embed, Embedder
from sycamore.execution.transforms.entity_extraction import ExtractEntity, EntityExtractor
from sycamore.execution.transforms.mapping import Filter
from sycamore.execution.transforms.partition import Partitioner
from sycamore.execution.transforms.summarize import Summarizer, Summarize
from sycamore.execution.transforms.table_extraction import TableExtractor
from sycamore.writer import DocSetWriter

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

    def show(
        self,
        limit: int = 20,
        show_elements: bool = True,
        show_binary: bool = False,
        truncate_content: bool = True,
        truncate_length: int = 100,
    ) -> None:
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan)
        documents = [Document(row) for row in dataset.take(limit)]
        for document in documents:
            if not show_elements:
                del document.elements
            if not show_binary:
                del document.binary_representation
            if truncate_content:
                if document.text_representation:
                    document.text_representation = document.text_representation[:truncate_length]
                if document.binary_representation:
                    document.binary_representation = document.binary_representation[:truncate_length]
                num_of_elements = len(document.elements)
                if num_of_elements > 0:
                    document.elements = document.elements[: min(truncate_length, num_of_elements)]
                if document.embedding is not None:
                    embedding_length = len(document.embedding)
                    document.embedding = document.embedding[: min(truncate_length, embedding_length)]

            pprint.pp(document, depth=2)

    def count(self) -> int:
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan)
        return dataset.count()

    def take(self, limit: int = 20) -> list[Document]:
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan)
        return [Document(row) for row in dataset.take(limit)]

    def limit(self, limit: int = 20) -> "DocSet":
        from sycamore.execution.transforms import Limit

        return DocSet(self.context, Limit(self.plan, limit))

    def partition(self, partitioner: Partitioner, table_extractor: Optional[TableExtractor] = None, **kwargs):
        plan = Partition(self.plan, partitioner=partitioner, table_extractor=table_extractor, **kwargs)
        return DocSet(self.context, plan)

    def explode(self, **resource_args):
        """Explode a list column into top level document

        To keep document has same schema, a document is

        Returns: A DocSet
        Each document has schema like below
        {"type": "pdf", "content": {"binary": xxx, "text": None},
         "doc_id": uuid, "parent_id": None, "properties": {
         "path": xxx, "author": "xxx", "title": "xxx"}}
        {"type": title, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-1, "parent_id": uuid},
        {"type": figure_caption, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-2, "parent_id": uuid},
        {"type": table, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-3, "parent_id": uuid},
        {"type": text, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-4, "parent_id": uuid},
        {"type": figure, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-5, "parent_id": uuid},
        {"type": table, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-6, "parent_id": uuid}
        """
        from sycamore.execution.transforms.explode import Explode

        explode = Explode(self.plan, **resource_args)
        return DocSet(self.context, explode)

    def embed(self, embedder: Embedder, **kwargs):
        embeddings = Embed(self.plan, embedder=embedder, **kwargs)
        return DocSet(self.context, embeddings)

    def extract_entity(self, entity_extractor: EntityExtractor, **kwargs) -> "DocSet":
        entities = ExtractEntity(self.plan, entity_extractor=entity_extractor, **kwargs)
        return DocSet(self.context, entities)

    def summarize(self, summarizer: Summarizer, **kwargs) -> "DocSet":
        summaries = Summarize(self.plan, summarizer=summarizer, **kwargs)
        return DocSet(self.context, summaries)

    def map(self, f: Callable[[Document], Document], **resource_args) -> "DocSet":
        from sycamore.execution.transforms.mapping import Map

        mapping = Map(self.plan, f=f, **resource_args)
        return DocSet(self.context, mapping)

    def flat_map(self, f: Callable[[Document], list[Document]], **resource_args) -> "DocSet":
        from sycamore.execution.transforms.mapping import FlatMap

        flat_map = FlatMap(self.plan, f=f, **resource_args)
        return DocSet(self.context, flat_map)

    def filter(self, f: Callable[[Document], bool], **resource_args) -> "DocSet":
        filtered = Filter(self.plan, f=f, **resource_args)
        return DocSet(self.context, filtered)

    def map_batch(
        self,
        f: Callable[[list[Document]], list[Document]],
        f_args: Optional[Iterable[Any]] = None,
        f_kwargs: Optional[dict[str, Any]] = None,
        f_constructor_args: Optional[Iterable[Any]] = None,
        f_constructor_kwargs: Optional[dict[str, Any]] = None,
        **resource_args
    ) -> "DocSet":
        from sycamore.execution.transforms.mapping import MapBatch

        map_batch = MapBatch(
            self.plan,
            f=f,
            f_args=f_args,
            f_kwargs=f_kwargs,
            f_constructor_args=f_constructor_args,
            f_constructor_kwargs=f_constructor_kwargs,
            **resource_args
        )
        return DocSet(self.context, map_batch)

    @property
    def write(self) -> DocSetWriter:
        return DocSetWriter(self.context, self.plan)
