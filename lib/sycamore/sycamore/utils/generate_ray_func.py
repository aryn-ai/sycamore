from typing import Any, Callable

import numpy as np

from sycamore.data import Document


def rename(new_function_name: str):
    def decorator(f):
        f.__name__ = new_function_name
        return f

    return decorator


def _get_documents_from_columnar_format(doc_batch: dict[str, np.ndarray]) -> list[Document]:
    return [Document(ser_doc) for ser_doc in doc_batch["doc"]]


def _get_columnar_format_from_documents(doc_batch: list[Document]) -> dict[str, list]:
    return {"doc": [doc.serialize() for doc in doc_batch]}


def generate_map_function(f: Callable[[Document], Document]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    @rename(f.__name__)
    def ray_callable(input_dict: dict[str, Any]) -> dict[str, Any]:
        document = f(Document.from_row(input_dict))
        return document.to_row()

    return ray_callable


def generate_map_class_from_callable(f: Callable[[Document], Document]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def ray_callable(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        document = f(Document.from_row(input_dict))
        return document.to_row()

    new_class = type(f.__name__, (), {"__call__": ray_callable})
    return new_class


def generate_map_batch_filter_class_from_callable(
    f: Callable[[Document], bool]
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def ray_callable(self, doc_batch: dict[str, np.ndarray]) -> dict[str, Any]:
        input_docs = _get_documents_from_columnar_format(doc_batch)
        output_docs = list(filter(f, input_docs))

        return _get_columnar_format_from_documents(output_docs)

    new_class = type("CustomRay", (), {"__call__": ray_callable})
    return new_class
