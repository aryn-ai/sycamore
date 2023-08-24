import logging
from typing import (Callable, List, Dict, Optional)

from pyarrow import Schema

from sycamore import Context
from sycamore.data import Document
from sycamore.execution import Node
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

    def show(self, limit: int = 20) -> None:
        from sycamore import Execution
        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan)
        for row in dataset.take(limit):
            print(row)

    @staticmethod
    def schema() -> "Schema":
        # TODO, enforce schema for document, also properties need to be
        #   convert to MapType?
        import pyarrow as pa
        return pa.schema([
            ('doc_id', pa.string()),
            ('type', pa.string()),
            ('content', pa.struct(
                [('binary', pa.large_binary()), ('text', pa.large_string())])),
            ('elements', pa.struct([('array', pa.large_list(pa.struct([
                ('type', pa.string()),
                ('content', pa.struct([
                    'binary', pa.large_binary(),
                    'text', pa.large_string()
                ])),
                ('properties', pa.map_(pa.string(), pa.string()))
            ])))])),
            ('embedding', pa.struct(
                [('binary', pa.large_list(pa.large_list(pa.float64()))),
                 ('text', pa.large_list(pa.large_list(pa.float64())))])),
            ('parent_id', pa.string()),
            ('properties', pa.map_(pa.string(), pa.string()))
        ])

    def unstructured_partition(self, **kwargs) -> "DocSet":
        """Partition pdf using unstructured library
        Returns: DocSet
        Each Document has schema like below
        {
            "content": {"binary": xxx, "text": None}
            "doc_id": uuid,
            "elements": {
                "array": [
                    {"type": title, "content": {"binary": "xxx"}, ...},
                    {"type": figure_caption, "content": {"text": "xxx"}},
                    {"type": table, "content": {"text": "xxx"}},
                    {"type": text, "content": {"text": "xxx"}},
                    ...
                ]
            }
            "properties": {
                "path": "xxx"
            }
        }
        """
        from sycamore.execution.transforms.partition import \
            UnstructuredPartition
        plan = UnstructuredPartition(self.plan, **kwargs)
        return DocSet(self.context, plan)

    def explode(self):
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
        explode = Explode(self.plan)
        return DocSet(self.context, explode)

    def sentence_transformer_embed(
            self,
            *,
            model_name: str,
            batch_size: int = None,
            device: str = None,
            **resource_args) -> "DocSet":
        """Embed using HuggingFace sentence transformer

        Args:
            model_name: model name to embed
            batch_size: batch size
            device: device needed
            **resource_args: resource related args

        Returns: A DocSet
        Each document has schema like below
        {"type": "pdf", "content": {"binary": xxx, "text": None},
         "doc_id": uuid, "parent_id": None, "properties": {
         "path": xxx, "author": "xxx", "title": "xxx"},
          "embedding": {"binary": xxx, "text": "xxx"}},
        {"type": title, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-1, "parent_id": uuid,
          "embedding": {"binary": xxx, "text": "xxx"}},
        {"type": figure_caption, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-2, "parent_id": uuid,
          "embedding": {"binary": xxx, "text": "xxx"}},
        {"type": table, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-3, "parent_id": uuid,
          "embedding": {"binary": xxx, "text": "xxx"}},
        {"type": text, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-4, "parent_id": uuid,
          "embedding": {"binary": xxx, "text": "xxx"}},
        {"type": figure, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-5, "parent_id": uuid,
          "embedding": {"binary": xxx, "text": "xxx"}},
        {"type": table, "content": {"binary": xxx, "text": None},
         "doc_id": uuid-6, "parent_id": uuid,
          "embedding": {"binary": xxx, "text": "xxx"}}
        """
        from sycamore.execution.transforms import SentenceTransformerEmbedding
        embedding = SentenceTransformerEmbedding(
            self.plan,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            **resource_args)
        return DocSet(self.context, embedding)

    def llm_entity_extract(self,
                           *,
                           entities_to_extract: Dict,
                           num_of_elements: int,
                           model_name: str,
                           model_args: Optional[Dict] = None,
                           **kwargs
                           ) -> "DocSet":
        from sycamore.execution.transforms import LLMEntityExtraction
        entities = LLMEntityExtraction(
            self.plan,
            entities_to_extract=entities_to_extract,
            num_of_elements=num_of_elements,
            model_name=model_name,
            model_args=model_args,
            **kwargs
        )
        return DocSet(self.context, entities)

    def map(self, f: Callable[[Document], Document]) -> "DocSet":
        from sycamore.execution.transforms.mapping import Map
        mapping = Map(self.plan, f=f)
        return DocSet(self.context, mapping)

    def flat_map(
            self,
            f: Callable[[Document], List[Document]],
            **kwargs) -> "DocSet":
        from sycamore.execution.transforms.mapping import FlatMap
        flat_map = FlatMap(self.plan, f=f, **kwargs)
        return DocSet(self.context, flat_map)

    def map_batch(
            self,
            f: Callable[[List[Document]], List[Document]],
            **kwargs) -> "DocSet":
        from sycamore.execution.transforms.mapping import MapBatch
        map_batch = MapBatch(self.plan, f=f, **kwargs)
        return DocSet(self.context, map_batch)

    @property
    def write(self) -> DocSetWriter:
        return DocSetWriter(self.context, self.plan)
