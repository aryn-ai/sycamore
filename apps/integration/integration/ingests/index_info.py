from dataclasses import dataclass


@dataclass
class IndexInfo:
    name: str
    num_docs: int
