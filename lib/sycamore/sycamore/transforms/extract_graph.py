from abc import ABC, abstractmethod

from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace

class GraphData(ABC):
    def __init__(self):
        pass

class GraphMetadata(GraphData):
    """
    Turn document metadata into nodes to put into a graph
    """
    def __init__(self, nodeKey: str, nodeLabel: str, relLabel: str):
        self.nodeKey = nodeKey
        self.nodeLabel = nodeLabel
        self.relLabel = relLabel
        self.labels = None

