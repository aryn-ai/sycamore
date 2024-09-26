from abc import ABC, abstractmethod
from typing import Optional, Union

from sycamore.data.document import Document, HierarchicalDocument
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map

import uuid


class DocumentStructure(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def extract(document: Document) -> Union[Document, HierarchicalDocument]:
        pass


class StructureByImages(DocumentStructure):
    @staticmethod
    def extract(doc: Document) -> HierarchicalDocument:
        doc = HierarchicalDocument(doc.data)
        doc.data["relationships"] = doc.get("relationships", {})
        doc.data["label"] = doc.get("label", "DOCUMENT")

        images = [child for child in doc.children if child.type == "Image"]
        doc.children = []
        for image in images:
            initial_page = HierarchicalDocument(
                {
                    "type": "Section-header",
                    "bbox": (0, 0, 0, 0),
                    "properties": {"score": 1, "page_number": 1},
                    "text_representation": "Document",
                    "binary_representation": b"Front Page",
                    "relationships": {},
                    "label": "SECTION",
                }
            )
            rel_1 = {
                "TYPE": "SECTION_OF",
                "properties": {},
                "START_ID": initial_page.doc_id,
                "START_LABEL": "SECTION",
                "END_ID": doc.doc_id,
                "END_LABEL": "DOCUMENT",
            }
            rel_2 = {
                "TYPE": "PART_OF",
                "properties": {},
                "START_ID": image.doc_id,
                "START_LABEL": "IMAGE",
                "END_ID": initial_page.doc_id,
                "END_LABEL": "SECTION",
            }
            doc.data["relationships"][str(uuid.uuid4())] = rel_1
            initial_page.data["relationships"][str(uuid.uuid4())] = rel_2
            initial_page.children.append(image)
            doc.children.append(initial_page)
        return ExtractImageSummaries.summarize(doc)


class StructureBySection(DocumentStructure):
    """
    Organizes documents by their section headers which encompass all of the elements
    between them and the next section header. Useful for long documents which have
    many elements that can't fit in an LLM's context window.
    """

    @staticmethod
    def extract(doc: Document) -> HierarchicalDocument:
        doc = HierarchicalDocument(doc.data)
        # if the first element is not a section header, insert generic placeholder
        if len(doc.children) > 0 and doc.children[0]["type"] != "Section-header":
            initial_page = HierarchicalDocument(
                {
                    "type": "Section-header",
                    "bbox": (0, 0, 0, 0),
                    "properties": {"score": 1, "page_number": 1},
                    "text_representation": "Front Page",
                    "binary_representation": b"Front Page",
                }
            )
            doc.children.insert(0, initial_page)  # O(n) insert :( we should use deque for everything

        doc.data["relationships"] = doc.get("relationships", {})
        doc.data["label"] = doc.get("label", "DOCUMENT")

        sections = []

        section: Optional[HierarchicalDocument] = None
        element: Optional[HierarchicalDocument] = None
        for child in doc.children:
            child.data["relationships"] = child.get("relationships", {})
            if child.type == "Section-header" and child.data.get("text_representation"):
                if section is not None:
                    next = {
                        "TYPE": "NEXT",
                        "properties": {},
                        "START_ID": section.doc_id,
                        "START_LABEL": "SECTION",
                        "END_ID": child.doc_id,
                        "END_LABEL": "SECTION",
                    }
                    child.data["relationships"][str(uuid.uuid4())] = next
                    element = None
                rel = {
                    "TYPE": "SECTION_OF",
                    "properties": {},
                    "START_ID": child.doc_id,
                    "START_LABEL": "SECTION",
                    "END_ID": doc.doc_id,
                    "END_LABEL": "DOCUMENT",
                }
                doc.data["relationships"][str(uuid.uuid4())] = rel
                child.data["label"] = "SECTION"
                section = child
                sections.append(section)
            else:
                assert section is not None
                if element is not None:
                    next = {
                        "TYPE": "NEXT",
                        "properties": {},
                        "START_ID": element.doc_id,
                        "START_LABEL": "ELEMENT",
                        "END_ID": child.doc_id,
                        "END_LABEL": "ELEMENT",
                    }
                    child.data["relationships"][str(uuid.uuid4())] = next
                rel = {
                    "TYPE": "PART_OF",
                    "properties": {},
                    "START_ID": child.doc_id,
                    "START_LABEL": "ELEMENT",
                    "END_ID": section.doc_id,
                    "END_LABEL": "SECTION",
                }
                section.data["relationships"][str(uuid.uuid4())] = rel
                child.data["label"] = "ELEMENT"
                element = child
                section.data["children"].append(element)

        doc.children = sections
        return ExtractTextSummaries.summarize(doc)


class StructureByDocument(DocumentStructure):
    """
    Organizes documents by using a single section to encompass all of a documents
    elements. Useful for short documents whose elements can be fit into an LLM's input
    context in a single shot.
    """

    @staticmethod
    def extract(doc: Document) -> HierarchicalDocument:
        doc = HierarchicalDocument(doc.data)
        doc.data["relationships"] = doc.get("relationships", {})
        doc.data["label"] = doc.get("label", "DOCUMENT")

        initial_page = HierarchicalDocument(
            {
                "type": "Section-header",
                "bbox": (0, 0, 0, 0),
                "properties": {"score": 1, "page_number": 1},
                "text_representation": "Document",
                "binary_representation": b"Front Page",
                "relationships": {},
                "label": "SECTION",
            }
        )

        rel = {
            "TYPE": "SECTION_OF",
            "properties": {},
            "START_ID": initial_page.doc_id,
            "START_LABEL": "SECTION",
            "END_ID": doc.doc_id,
            "END_LABEL": "DOCUMENT",
        }
        initial_page.data["relationships"][str(uuid.uuid4())] = rel

        section: HierarchicalDocument = initial_page
        element: Optional[HierarchicalDocument] = None
        for child in doc.children:
            child.data["relationships"] = child.get("relationships", {})
            if element is not None:
                next = {
                    "TYPE": "NEXT",
                    "properties": {},
                    "START_ID": element.doc_id,
                    "START_LABEL": "ELEMENT",
                    "END_ID": child.doc_id,
                    "END_LABEL": "ELEMENT",
                }
                child.data["relationships"][str(uuid.uuid4())] = next
            rel = {
                "TYPE": "PART_OF",
                "properties": {},
                "START_ID": child.doc_id,
                "START_LABEL": "ELEMENT",
                "END_ID": section.doc_id,
                "END_LABEL": "SECTION",
            }
            child.data["relationships"][str(uuid.uuid4())] = rel
            child.data["label"] = "ELEMENT"
            element = child
            section.data["children"].append(element)

        doc.children = [section]
        return ExtractTextSummaries.summarize(doc)


class ExtractDocumentStructure(Map):
    """
    Transforms a document into a Hierarchical document defining its hierarchy from
    rules defined by the passed in DocumentStructure class. Additionally, adds
    uuid's and relationships between hierarchical nodes so that the document can
    be loaded into neo4j.
    """

    def __init__(
        self,
        child: Node,
        structure: DocumentStructure,
        **resource_args,
    ):
        super().__init__(child, f=structure.extract, **resource_args)


class ExtractSummary(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def summarize(doc: HierarchicalDocument) -> HierarchicalDocument:
        pass


class ExtractTextSummaries(ExtractSummary):
    @staticmethod
    def summarize(doc: HierarchicalDocument) -> HierarchicalDocument:
        for section in doc.children:
            assert section.text_representation is not None
            summary_list = []
            sec_sum = f"-----SECTION TITLE: {section.text_representation.strip()}-----\n"
            summary_list.append(sec_sum)
            for element in section.children:
                if element.type is not None and element.text_representation is not None
                    elem_sum = f"---Element Type: {element.type.strip()}---\n{element.text_representation.strip()}\n"
                    summary_list.append(elem_sum)
            section.data["summary"] = "".join(summary_list)
        return doc


class ExtractImageSummaries(ExtractSummary):
    @staticmethod
    def summarize(doc: HierarchicalDocument) -> HierarchicalDocument:
        for section in doc.children:
            assert len(section.children) == 1 and section.children[0].type == "Image"
            image = section.children[0]
            section.data["summary"] = image.data
        return doc
