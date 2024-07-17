from abc import ABC


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
