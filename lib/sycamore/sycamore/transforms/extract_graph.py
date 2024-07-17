from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sycamore.docset import DocSet


class GraphData(ABC):
    def __init__(self):
        pass


class GraphMetadata(GraphData):
    """
    Args:
        nodeKey: Key used to access document metadata in the document['properties] dictionary
        nodeLabel: The label used in neo4j the node of a piece of metadata
        relLabel: The label used in neo4j for the relationship between the document and a piece of metadata
    """

    def __init__(self, nodeKey: str, nodeLabel: str, relLabel: str):
        self.nodeKey = nodeKey
        self.nodeLabel = nodeLabel
        self.relLabel = relLabel

class GraphExtractor(ABC):
    def __init__(self, entity_name: str):
        self._entity_name = entity_name

    @abstractmethod
    def  extract(self, document: "DocSet") -> "DocSet":
        pass

class MetadataExtractor(GraphExtractor):
    """
    Extracts metadata from documents and represents them as nodes and relationship in neo4j

    Args:
        metadata: a list of GraphMetadata that is used to determine what metadata is extracted
    """
    def __init__(self, metadata: list[GraphMetadata]):
        self.metadata = metadata            


    #no parallization
    def extract(self, docset: "DocSet") -> "DocSet":
        from sycamore import Execution
        from sycamore.data import Document, MetadataDocument
        from sycamore.reader import DocSetReader
        from collections import defaultdict
        import uuid

        nodes = defaultdict(lambda: defaultdict(dict))
        execution = Execution(docset.context, docset.plan)
        dataset = execution.execute(docset.plan)
        all_docs = [Document.from_row(row) for row in dataset.take_all(None)]
        docs = [d for d in all_docs if not isinstance(d, MetadataDocument)]
        for doc in docs:
            for m in self.metadata:
                value = doc["properties"][m.nodeKey]
                if nodes[m.nodeKey][value] == {}:
                    nodes[m.nodeKey][value] = {
                        "doc_id": str(uuid.uuid4()),
                        "label": str(m.nodeLabel),
                        "type": "metadata",
                        "relationships": {},
                        "properties": {str(m.nodeKey): str(value)},
                    }
                rel = {
                    "START_ID": str(doc.doc_id),
                    "END_ID": str(nodes[m.nodeKey][value]["doc_id"]),
                    "TYPE": str(m.relLabel),
                    "properties": {},
                }
                nodes[m.nodeKey][value]["relationships"][str(uuid.uuid4())] = rel

        # docset must be larger than size 0
        for label in nodes.values():
            for value in label.values():
                docs[0]["elements"].append(value)

        reader = DocSetReader(docset.context)
        return reader.document(docs)
