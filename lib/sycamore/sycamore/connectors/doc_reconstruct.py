from typing import Callable

from sycamore.data import Document


class DocumentReconstructor:
    def __init__(self, index_name: str, reconstruct_fn: Callable[[str, str], Document]):
        self.index_name = index_name
        self.reconstruct_fn = reconstruct_fn

    def get_required_source_fields(self) -> list[str]:
        return ["parent_id"]

    def get_doc_id(self, data: dict) -> str:
        return data["_source"]["parent_id"] or data["_id"]

    def reconstruct(self, data: dict) -> Document:
        return self.reconstruct_fn(self.index_name, self.get_doc_id(data))
