from enum import Enum
import logging
import pprint
from typing import TYPE_CHECKING

from sycamore.plan_nodes import Node, UnaryNode
from sycamore.data import Document, MetadataDocument

if TYPE_CHECKING:
    from ray import Dataset


class MaterializeMode(Enum):
    UNKNOWN = 0
    INMEM_VERIFY_ONLY = 1
    # todo: more modes


class Materialize(UnaryNode):
    def __init__(self, child: Node, **kwargs):
        assert isinstance(child, Node)
        super().__init__(child, **kwargs)

    def execute(self, **kwargs) -> "Dataset":
        input_dataset = self.child().execute(**kwargs)
        md = []
        for row in input_dataset.iter_rows():
            doc = Document.from_row(row)
            if not isinstance(doc, MetadataDocument):
                continue
            md.append(doc)
        return input_dataset

    def local_execute(self, docs: list[Document]) -> list[Document]:
        md = [d for d in docs if isinstance(d, MetadataDocument)]
        logging.info(f"Found {len(md)} md documents")
        logging.info(f"\n{pprint.pformat(md)}")
        return docs
