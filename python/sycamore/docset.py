import logging
from typing import Callable, List, Dict, Optional, Union

from sycamore import Context
from sycamore.data import Document, Element
from sycamore.execution import Node
from sycamore.execution.transforms import LLMExtractEntity, PartitionerOptions
from sycamore.execution.transforms.entity import EntityExtractor
from sycamore.execution.transforms.llms import LLM
from sycamore.execution.transforms.summarize import SummarizeText
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

    def count(self) -> int:
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan)
        return dataset.count()

    def take(self, limit: int = 20) -> List[Document]:
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan)
        return [Document(row) for row in dataset.take(limit)]

    def partition(self, options: PartitionerOptions, **resource_args) -> "DocSet":
        """Partition document using unstructured library
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
        from sycamore.execution.transforms.partition import Partition

        plan = Partition(self.plan, options, **resource_args)
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

    def sentence_transformer_embed(
        self, *, model_name: str, batch_size: int = None, device: str = None, **resource_args
    ) -> "DocSet":
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
            self.plan, model_name=model_name, batch_size=batch_size, device=device, **resource_args
        )
        return DocSet(self.context, embedding)

    def llm_extract_entity(
        self,
        *,
        entity_to_extract: Union[str, Dict],
        num_of_elements: int = 10,
        llm: LLM,
        prompt_template: str = None,
        prompt_formatter: Callable[[List[Element]], str] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        **kwargs
    ) -> "DocSet":
        entities = LLMExtractEntity(
            self.plan,
            entity_to_extract=entity_to_extract,
            num_of_elements=num_of_elements,
            llm=llm,
            prompt_template=prompt_template,
            prompt_formatter=prompt_formatter,
            entity_extractor=entity_extractor,
            **kwargs
        )
        return DocSet(self.context, entities)

    def extract_tables(
        self, profile_name: str = None, region_name: str = None, kms_key_id: str = "", **kwargs
    ) -> "DocSet":
        from sycamore.execution.transforms import TableExtraction

        table_extraction = TableExtraction(self.plan, profile_name, region_name, kms_key_id, **kwargs)
        return DocSet(self.context, table_extraction)

    def summarize(
        self, *, llm: LLM, element_operator: Callable[[Document], List[Element]] = None, **kwargs
    ) -> "DocSet":
        summaries = SummarizeText(self.plan, llm=llm, element_operator=element_operator, **kwargs)
        return DocSet(self.context, summaries)

    def map(self, f: Callable[[Document], Document]) -> "DocSet":
        from sycamore.execution.transforms.mapping import Map

        mapping = Map(self.plan, f=f)
        return DocSet(self.context, mapping)

    def flat_map(self, f: Callable[[Document], List[Document]], **kwargs) -> "DocSet":
        from sycamore.execution.transforms.mapping import FlatMap

        flat_map = FlatMap(self.plan, f=f, **kwargs)
        return DocSet(self.context, flat_map)

    def map_batch(self, f: Callable[[List[Document]], List[Document]], **kwargs) -> "DocSet":
        from sycamore.execution.transforms.mapping import MapBatch

        map_batch = MapBatch(self.plan, f=f, **kwargs)
        return DocSet(self.context, map_batch)

    @property
    def write(self, **resource_args) -> DocSetWriter:
        return DocSetWriter(self.context, self.plan, **resource_args)
