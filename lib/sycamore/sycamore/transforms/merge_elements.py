from abc import ABC, abstractmethod
from typing import Any, Dict


from sycamore.data import Document, Element, BoundingBox
from sycamore.plan_nodes import SingleThreadUser, NonGPUUser, Node
from sycamore.functions.tokenizer import Tokenizer
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace


class ElementMerger(ABC):
    @abstractmethod
    def should_merge(self, element1: Element, element2: Element) -> bool:
        pass

    @abstractmethod
    def merge(self, element1: Element, element2: Element) -> Element:
        pass

    @abstractmethod
    def preprocess_element(self, element: Element) -> Element:
        pass

    @abstractmethod
    def postprocess_element(self, element: Element) -> Element:
        pass

    @timetrace("mergeElem")
    def merge_elements(self, document: Document) -> Document:
        """Use self.should_merge and self.merge to greedily merge consecutive elements.
        If the next element should be merged into the last 'accumulation' element, merge it.

        Args:
            document (Document): A document with elements to be merged.

        Returns:
            Document: The same document, with its elements merged
        """
        if len(document.elements) < 2:
            return document
        to_merge = [self.preprocess_element(e) for e in document.elements]
        new_elements = [to_merge[0]]
        for element in to_merge[1:]:
            if self.should_merge(new_elements[-1], element):
                new_elements[-1] = self.merge(new_elements[-1], element)
            else:
                new_elements.append(element)
        document.elements = [self.postprocess_element(e) for e in new_elements]
        return document


class GreedyTextElementMerger(ElementMerger):
    """
    The ``GreedyTextElementMerger`` takes a tokenizer and a token limit, and merges elements together,
    greedily, until the combined element will overflow the token limit, at which point the merger
    starts work on a new merged element. If an element is already too big, the `GreedyTextElementMerger`
    will leave it alone.
    """

    def __init__(self, tokenizer: Tokenizer, max_tokens: int, merge_across_pages: bool = True):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.merge_across_pages = merge_across_pages

    def preprocess_element(self, element: Element) -> Element:
        element.data["token_count"] = len(self.tokenizer.tokenize(element.text_representation or ""))
        return element

    def postprocess_element(self, element: Element) -> Element:
        del element.data["token_count"]
        return element

    def should_merge(self, element1: Element, element2: Element) -> bool:
        if not self.merge_across_pages and element1.properties["page_number"] != element2.properties["page_number"]:
            return False
        if element1.data["token_count"] + 1 + element2.data["token_count"] > self.max_tokens:
            return False
        return True

    def merge(self, elt1: Element, elt2: Element) -> Element:
        """Merge two elements; the new element's fields will be set as:
            - type: "Section"
            - binary_representation: elt1.binary_representation + elt2.binary_representation
            - text_representation: elt1.text_representation + elt2.text_representation
            - bbox: the minimal bbox that contains both elt1's and elt2's bboxes
            - properties: elt1's properties + any of elt2's properties that are not in elt1
            note: if elt1 and elt2 have different values for the same property, we take elt1's value
            note: if any input field is None we take the other element's field without merge logic

        Args:
            element1 (Tuple[Element, int]): the first element (and number of tokens in it)
            element2 (Tuple[Element, int]): the second element (and number of tokens in it)

        Returns:
            Tuple[Element, int]: a new merged element from the inputs (and number of tokens in it)
        """
        tok1 = elt1.data["token_count"]
        tok2 = elt2.data["token_count"]
        new_elt = Element()
        new_elt.type = "Section"
        # Merge binary representations by concatenation
        if elt1.binary_representation is None or elt2.binary_representation is None:
            new_elt.binary_representation = elt1.binary_representation or elt2.binary_representation
        else:
            new_elt.binary_representation = elt1.binary_representation + elt2.binary_representation
        # Merge text representations by concatenation with a newline
        if elt1.text_representation is None or elt2.text_representation is None:
            new_elt.text_representation = elt1.text_representation or elt2.text_representation
            new_elt.data["token_count"] = max(tok1, tok2)
        else:
            new_elt.text_representation = elt1.text_representation + "\n" + elt2.text_representation
            new_elt.data["token_count"] = tok1 + 1 + tok2
        # Merge bbox by taking the coords that make the largest box
        if elt1.bbox is None and elt2.bbox is None:
            pass
        elif elt1.bbox is None or elt2.bbox is None:
            new_elt.bbox = elt1.bbox or elt2.bbox
        else:
            new_elt.bbox = BoundingBox(
                min(elt1.bbox.x1, elt2.bbox.x1),
                min(elt1.bbox.y1, elt2.bbox.y1),
                max(elt1.bbox.x2, elt2.bbox.x2),
                max(elt1.bbox.y2, elt2.bbox.y2),
            )
        # Merge properties by taking the union of the keys
        properties = new_elt.properties
        for k, v in elt1.properties.items():
            properties[k] = v
            if k == "page_number":
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))
        for k, v in elt2.properties.items():
            if properties.get(k) is None:
                properties[k] = v
            # if a page number exists, add it to the set of page numbers for this new element
            if k == "page_number":
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))

        new_elt.properties = properties

        return new_elt


class GreedySectionMerger(ElementMerger):
    """
    The ``GreedySectionMerger`` groups together different elements in a Document according to three rules. All rules
    are subject to the max_tokens limit and merge_across_pages flag.

    - It merges adjacent text elements.
    - It merges an adjacent Section-header and an image. The new element type is called Section-header+image.
    - It merges an Image and subsequent adjacent text elements.
    """

    def __init__(self, tokenizer: Tokenizer, max_tokens: int, merge_across_pages: bool = True):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.merge_across_pages = merge_across_pages

    def preprocess_element(self, element: Element) -> Element:
        if element.type == "Image" and "summary" in element.properties and "summary" in element.properties["summary"]:
            element.data["token_count"] = len(self.tokenizer.tokenize(element.properties["summary"]["summary"] or ""))
        else:
            element.data["token_count"] = len(self.tokenizer.tokenize(element.text_representation or ""))
        return element

    def postprocess_element(self, element: Element) -> Element:
        del element.data["token_count"]
        return element

    def should_merge(self, element1: Element, element2: Element) -> bool:
        # deal with empty elements
        if (
            "page_number" not in element1.properties
            or "page_number" not in element2.properties
            or element1.type is None
            or element2.type is None
        ):
            return False

        # DO NOT MERGE across pages
        if not self.merge_across_pages and element1.properties["page_number"] != element2.properties["page_number"]:
            return False

        if element1.data["token_count"] + 1 + element2.data["token_count"] > self.max_tokens:
            return False

        # MERGE adjacent 'text' elements (but not across pages - see above)
        if (
            (element1 is not None)
            and (element1.type == "Text")
            and (element2 is not None)
            and (element2.type == "Text")
        ):
            return True

        # MERGE 'Section-header' + 'table'
        if (
            (element1 is not None)
            and (element1.type == "Section-header")
            and (element2 is not None)
            and (element2.type == "table")
        ):
            return True

        # Merge 'image' + 'text'* until next 'image' or 'Section-header'
        if (
            (element1 is not None)
            and (element1.type == "Image" or element1.type == "Image+Text")
            and (element2 is not None)
            and (element2.type == "Text")
        ):
            return True
        return False

    def merge(self, elt1: Element, elt2: Element) -> Element:
        """Merge two elements; the new element's fields will be set as:
            - type: "Section"
            - binary_representation: elt1.binary_representation + elt2.binary_representation
            - text_representation: elt1.text_representation + elt2.text_representation
            - bbox: the minimal bbox that contains both elt1's and elt2's bboxes
            - properties: elt1's properties + any of elt2's properties that are not in elt1
            note: if elt1 and elt2 have different values for the same property, we take elt1's value
            note: if any input field is None we take the other element's field without merge logic

        Args:
            element1 (Tuple[Element, int]): the first element (and number of tokens in it)
            element2 (Tuple[Element, int]): the second element (and number of tokens in it)

        Returns:
            Tuple[Element, int]: a new merged element from the inputs (and number of tokens in it)
        """

        tok1 = elt1.data["token_count"]
        tok2 = elt2.data["token_count"]
        new_elt = Element()
        # 'text' + 'text' = 'text'
        # 'image' + 'text' = 'image+text'
        # 'image+text' + 'text' = 'image+text'
        # 'Section-header' + 'table' = 'Section-header+table'

        if (elt1.type == "Image" or elt1.type == "Image+Text") and elt2.type == "Text":
            new_elt.type = "Image+Text"
        elif elt1.type == "Text" and elt2.type == "Text":
            new_elt.type = "Text"
        elif elt1.type == "Section-header" and elt2.type == "table":
            new_elt.type = "Section-header+table"
        else:
            new_elt.type = "????"

        # Merge binary representations by concatenation
        if elt1.binary_representation is None or elt2.binary_representation is None:
            new_elt.binary_representation = elt1.binary_representation or elt2.binary_representation
        else:
            new_elt.binary_representation = elt1.binary_representation + elt2.binary_representation

        # Merge text representations by concatenation with a newline
        if elt1.type != "Image" and (elt1.text_representation is None or elt2.text_representation is None):
            new_elt.text_representation = elt1.text_representation or elt2.text_representation
            new_elt.data["token_count"] = max(tok1, tok2)

        else:
            if new_elt.type == "Image+Text":
                # text rep = summary(image) + text
                if elt1.type == "Image":
                    if "summary" in elt1.properties and "summary" in elt1.properties["summary"]:
                        new_elt.text_representation = (
                            elt1.properties["summary"]["summary"] + "\n" + elt2.text_representation
                        )
                    else:
                        new_elt.text_representation = elt2.text_representation
                else:
                    if elt1.text_representation and elt2.text_representation:
                        new_elt.text_representation = elt1.text_representation + "\n" + elt2.text_representation
                    else:
                        new_elt.text_representation = elt2.text_representation
                new_elt.data["token_count"] = tok1 + 1 + tok2

            elif new_elt.type == "Section-header+table":
                # text rep = header text + table html
                if hasattr(elt2, "table") and elt2.table:
                    if elt1.text_representation:
                        new_elt.text_representation = elt1.text_representation + "\n" + elt2.table.to_html()
                        new_elt.data["token_count"] = tok1 + 1 + tok2
                else:
                    new_elt.text_representation = elt1.text_representation
                    new_elt.data["token_count"] = tok1
            else:
                # text + text
                new_elt_text_representation = "\n".join(
                    filter(None, [elt1.text_representation, elt2.text_representation])
                )
                new_elt.text_representation = new_elt_text_representation if new_elt_text_representation else None
                # new_elt.text_representation = elt1.text_representation + "\n" + elt2.text_representation
                new_elt.data["token_count"] = tok1 + 1 + tok2

        # Merge bbox by taking the coords that make the largest box
        if elt1.bbox is None and elt2.bbox is None:
            pass
        elif elt1.bbox is None or elt2.bbox is None:
            new_elt.bbox = elt1.bbox or elt2.bbox
        else:
            new_elt.bbox = BoundingBox(
                min(elt1.bbox.x1, elt2.bbox.x1),
                min(elt1.bbox.y1, elt2.bbox.y1),
                max(elt1.bbox.x2, elt2.bbox.x2),
                max(elt1.bbox.y2, elt2.bbox.y2),
            )

        # Merge properties by taking the union of the keys
        properties = new_elt.properties
        for k, v in elt1.properties.items():
            properties[k] = v
            if k == "page_number":
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))
        for k, v in elt2.properties.items():
            if properties.get(k) is None:
                properties[k] = v
            # if a page number exists, add it to the set of page numbers for this new element
            if k == "page_number":
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))

        new_elt.properties = properties

        return new_elt


class MarkedMerger(ElementMerger):
    """
    The ``MarkedMerger`` merges elements by referencing "marks" placed on the elements by the transforms
    in ``sycamore.transforms.bbox_merge`` and ``sycamore.transforms.mark_misc``. The marks are "_break"
    and "_drop". The `MarkedMerger` will merge elements until it hits a "_break" mark, whereupon it will
    start a new element. It handles elements marked with "_drop" by, well, dropping them entirely.
    """

    def should_merge(self, element1: Element, element2: Element) -> bool:
        return False

    def merge(self, element1: Element, element2: Element) -> Element:
        return element1

    def preprocess_element(self, elem: Element) -> Element:
        return elem

    def postprocess_element(self, elem: Element) -> Element:
        return elem

    @timetrace("mergeMarked")
    def merge_elements(self, document: Document) -> Document:
        if len(document.elements) < 1:
            return document

        # merge elements, honoring marked breaks and drops
        merged = []
        bin = b""
        text = ""
        props: Dict[str, Any] = {}
        bbox = None

        for elem in document.elements:
            if elem.data.get("_drop"):
                continue
            if elem.data.get("_break"):
                ee = Element()
                ee.binary_representation = bin
                ee.text_representation = text
                ee.properties = props
                ee.data["bbox"] = bbox
                merged.append(ee)
                bin = b""
                text = ""
                props = {}
                bbox = None
            if elem.binary_representation:
                bin += elem.binary_representation + b"\n"
            if elem.text_representation:
                text += elem.text_representation + "\n"
            for k, v in elem.properties.items():
                if k == "page_number":
                    props["page_numbers"] = props.get("page_numbers", list())
                    props["page_numbers"] = list(set(props["page_numbers"] + [v]))
                if k not in props:  # ??? order may matter here
                    props[k] = v
            ebb = elem.data.get("bbox")
            if ebb is not None:
                if bbox is None:
                    bbox = ebb
                else:
                    bbox = (min(bbox[0], ebb[0]), min(bbox[1], ebb[1]), max(bbox[2], ebb[2]), max(bbox[3], ebb[3]))

        if text:
            ee = Element()
            ee.binary_representation = bin
            ee.text_representation = text
            ee.properties = props
            ee.data["bbox"] = bbox
            merged.append(ee)

        document.elements = merged
        return document


class Merge(SingleThreadUser, NonGPUUser, Map):
    """
    Merge Elements into fewer large elements
    """

    def __init__(self, child: Node, merger: ElementMerger, **kwargs):
        super().__init__(child, f=merger.merge_elements, **kwargs)
