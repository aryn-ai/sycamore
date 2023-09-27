import logging
import pprint
import sys
from typing import Callable, Optional, Any, Iterable

from sycamore import Context
from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms.embed import Embedder
from sycamore.transforms.extract_entity import EntityExtractor
from sycamore.transforms.partition import Partitioner
from sycamore.transforms.summarize import Summarizer
from sycamore.transforms.extract_table import TableExtractor
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
        num_elements: int = -1,  # -1 shows all elements
        show_binary: bool = False,
        show_embedding: bool = False,
        truncate_content: bool = True,
        truncate_length: int = 100,
        stream=sys.stdout,
    ) -> None:
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan)
        documents = [Document(row) for row in dataset.take(limit)]
        for document in documents:
            if not show_elements:
                num_elems = len(document.elements)
                document.data["elements"]["array"] = f"<{num_elems} elements>"

            if not show_binary and document.binary_representation is not None:
                binary_length = len(document.binary_representation)
                document.binary_representation = f"<{binary_length} bytes>".encode("utf-8")

            if truncate_content and document.text_representation is not None:
                amount_truncated = len(document.text_representation) - truncate_length
                if amount_truncated > 0:
                    document.text_representation = (
                        document.text_representation[:truncate_length] + f" <{amount_truncated} chars>"
                    )

            if document.elements is not None and num_elements >= 0 and len(document.elements) > num_elements:
                document.elements = document.elements[:num_elements]

            if not show_embedding and document.embedding is not None:
                embedding_length = len(document.embedding)
                document.data["embedding"] = f"<{embedding_length} floats>"

            pprint.pp(document, stream=stream)

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
        from sycamore.transforms import Limit

        return DocSet(self.context, Limit(self.plan, limit))

    def partition(self, partitioner: Partitioner, table_extractor: Optional[TableExtractor] = None, **kwargs):
        from sycamore.transforms import Partition

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
        from sycamore.transforms.explode import Explode

        explode = Explode(self.plan, **resource_args)
        return DocSet(self.context, explode)

    def embed(self, embedder: Embedder, **kwargs):
        from sycamore.transforms import Embed

        embeddings = Embed(self.plan, embedder=embedder, **kwargs)
        return DocSet(self.context, embeddings)

    def extract_entity(self, entity_extractor: EntityExtractor, **kwargs) -> "DocSet":
        from sycamore.transforms import ExtractEntity

        entities = ExtractEntity(self.plan, entity_extractor=entity_extractor, **kwargs)
        return DocSet(self.context, entities)

    def summarize(self, summarizer: Summarizer, **kwargs) -> "DocSet":
        from sycamore.transforms import Summarize

        summaries = Summarize(self.plan, summarizer=summarizer, **kwargs)
        return DocSet(self.context, summaries)

    def map(self, f: Callable[[Document], Document], **resource_args) -> "DocSet":
        from sycamore.transforms import Map

        mapping = Map(self.plan, f=f, **resource_args)
        return DocSet(self.context, mapping)

    def flat_map(self, f: Callable[[Document], list[Document]], **resource_args) -> "DocSet":
        from sycamore.transforms import FlatMap

        flat_map = FlatMap(self.plan, f=f, **resource_args)
        return DocSet(self.context, flat_map)

    def filter(self, f: Callable[[Document], bool], **resource_args) -> "DocSet":
        from sycamore.transforms import Filter

        filtered = Filter(self.plan, f=f, **resource_args)
        return DocSet(self.context, filtered)

    def map_batch(
        self,
        f: Callable[[list[Document]], list[Document]],
        f_args: Optional[Iterable[Any]] = None,
        f_kwargs: Optional[dict[str, Any]] = None,
        f_constructor_args: Optional[Iterable[Any]] = None,
        f_constructor_kwargs: Optional[dict[str, Any]] = None,
        **resource_args,
    ) -> "DocSet":
        from sycamore.transforms import MapBatch

        map_batch = MapBatch(
            self.plan,
            f=f,
            f_args=f_args,
            f_kwargs=f_kwargs,
            f_constructor_args=f_constructor_args,
            f_constructor_kwargs=f_constructor_kwargs,
            **resource_args,
        )
        return DocSet(self.context, map_batch)

    @property
    def write(self) -> DocSetWriter:
        return DocSetWriter(self.context, self.plan)
