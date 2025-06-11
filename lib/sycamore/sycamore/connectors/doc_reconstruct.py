from typing import Callable

from sycamore.data import Document, Element
from sycamore.data.document import DocumentPropertyTypes, DocumentSource


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


class RAGDocumentReconstructor(DocumentReconstructor):
    def __init__(
        self,
        index_name: str,
        reconstruct_fn: Callable[[str, str], Document] = None,
    ):
        super().__init__(index_name, reconstruct_fn)

    def reconstruct(self, output: list[dict]) -> list[Document]:
        result: list[Document] = []
        unique_docs: dict[str, Document] = {}

        for element in output:
            doc = Document(
                {
                    **element.get("_source", {}),
                }
            )
            if not doc.parent_id:
                continue  # Skip elements without a parent_id
            doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
            unique_docs[doc.parent_id] = unique_docs.get(
                doc.parent_id,
                Document(
                    {
                        "doc_id": doc.parent_id,
                        "properties": {
                            **doc.properties,
                            DocumentPropertyTypes.SOURCE: DocumentSource.DOCUMENT_RECONSTRUCTION_PARENT,
                        },
                        "type": doc.type,
                    }
                ),
            )
            parent = unique_docs[doc.parent_id]
            parent.elements.append(Element(doc.data))

        result = list(unique_docs.values())
        return result
