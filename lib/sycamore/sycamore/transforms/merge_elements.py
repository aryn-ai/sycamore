from abc import ABC, abstractmethod
from typing import Any, Dict, Pattern, Optional
from collections import defaultdict
import re


from sycamore.data import Document, Element, BoundingBox, Table, TableElement, TableCell
from sycamore.data.document import DocumentPropertyTypes
from sycamore.plan_nodes import SingleThreadUser, NonGPUUser, Node
from sycamore.functions.tokenizer import Tokenizer
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
from sycamore.transforms.llm_query import LLMTextQueryAgent
from sycamore.llms import LLM
from sycamore.utils.bbox_sort import bbox_sort_document


class ElementMerger(ABC):
    @abstractmethod
    def should_merge(self, element1: Element, element2: Element) -> bool:
        pass

    @abstractmethod
    def merge(self, element1: Element, element2: Element) -> Element:
        pass

    def preprocess_element(self, element: Element) -> Element:
        return element

    def postprocess_element(self, element: Element) -> Element:
        return element

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
        if (
            not self.merge_across_pages
            and element1.properties[DocumentPropertyTypes.PAGE_NUMBER]
            != element2.properties[DocumentPropertyTypes.PAGE_NUMBER]
        ):
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
            if k == DocumentPropertyTypes.PAGE_NUMBER:
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))
        for k, v in elt2.properties.items():
            if properties.get(k) is None:
                properties[k] = v
            # if a page number exists, add it to the set of page numbers for this new element
            if k == DocumentPropertyTypes.PAGE_NUMBER:
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
            DocumentPropertyTypes.PAGE_NUMBER not in element1.properties
            or DocumentPropertyTypes.PAGE_NUMBER not in element2.properties
            or element1.type is None
            or element2.type is None
        ):
            return False

        # DO NOT MERGE across pages
        if (
            not self.merge_across_pages
            and element1.properties[DocumentPropertyTypes.PAGE_NUMBER]
            != element2.properties[DocumentPropertyTypes.PAGE_NUMBER]
        ):
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
                    if elt1.text_representation is not None:
                        new_elt.text_representation = elt1.text_representation + "\n" + elt2.table.to_html()
                        new_elt.data["token_count"] = tok1 + 1 + tok2
                    else:
                        new_elt.text_representation = elt2.table.to_html()
                        new_elt.data["token_count"] = tok2
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
            if k == DocumentPropertyTypes.PAGE_NUMBER:
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))
        for k, v in elt2.properties.items():
            if properties.get(k) is None:
                properties[k] = v
            # if a page number exists, add it to the set of page numbers for this new element
            if k == DocumentPropertyTypes.PAGE_NUMBER:
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
                if k == DocumentPropertyTypes.PAGE_NUMBER:
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


class TableMerger(ElementMerger):
    """
    The ``Table merger`` handles 3 operations
    1. If a text element (Caption, Section-header, Text...) contains the regex pattern anywhere in a page
     it is attached to the text_representation of the table on the page.
    2. LLMQuery is used for adding a table_continuation property to table elements. Is the table is
     a continuation from a previous table the property is stored as true, else false.
    3. After LLMQuery, table elements which are continuations are merged as one element.
    Example:
         .. code-block:: python

            llm = OpenAI(OpenAIModels.GPT_4O, api_key = '')

            prompt = "Analyze two CSV tables that may be parts of a single table split across pages. Determine\
            if the second table is a continuation of the first with 100% certainty. Check either of the following:\
            1. Column headers: Must be near identical in terms of text(the ordering/text may contain minor errors \
            because of OCR quality) in both tables. If the headers are almost the same check the number of columns,\
                 they should be roughly the same. \
            2. Missing headers: If the header/columns in the second table are missing, then the first row in the
            second table should logically be in continutaion of the last row in the first table.\
            Respond with only 'true' or 'false' based on your certainty that the second table is a continuation. \
            Certainty is determined if either of the two conditions is true."

            regex_pattern = r"table \\d+"

            merger = TableMerger(llm_prompt = prompt, llm=llm)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf", regex_pattern= regex_pattern)
                .partition(partitioner=ArynPartitioner())
                .merge(merger=merger)
    """

    def __init__(
        self,
        regex_pattern: Optional[Pattern] = None,
        llm_prompt: Optional[str] = None,
        llm: Optional[LLM] = None,
        *args,
        **kwargs,
    ):
        self.regex_pattern = regex_pattern
        self.llm_prompt = llm_prompt
        self.llm = llm

    def merge_elements(self, document: Document) -> Document:

        table_elements = [ele for ele in document.elements if ele.type == "table"]
        if len(table_elements) < 1:
            return document
        if self.regex_pattern:
            document.elements = self.customTableHeaderAdditionFilter(document.elements)
        if not self.llm_prompt or len(table_elements) < 2:
            return document
        document = self.process_llm_query(document)
        table_elements = [ele for ele in document.elements if ele.type == "table"]
        other_elements = [ele for ele in document.elements if ele.type != "table"]
        new_table_elements = [table_elements[0]]
        for element in table_elements[1:]:
            if self.should_merge(new_table_elements[-1], element):
                new_table_elements[-1] = self.merge(new_table_elements[-1], element)
                new_table_elements[-1]["properties"]["table_continuation"] = True
            else:
                new_table_elements.append(element)
                new_table_elements[-1]["properties"]["table_continuation"] = False
        other_elements.extend(new_table_elements)
        document.elements = other_elements
        bbox_sort_document(document)

        return document

    def should_merge(self, element1: Element, element2: Element) -> bool:
        if "table_continuation" in element2["properties"]:
            return "true" in element2["properties"]["table_continuation"].lower()
        return False

    def merge(self, elt1: Element, elt2: Element) -> Element:

        # Check if both elements are TableElements
        if not isinstance(elt1, TableElement) or not isinstance(elt2, TableElement):
            raise TypeError("Both elements must be of type TableElement to perform merging.")
        # Combine the cells, adjusting the row indices for the second table
        if elt1.table is None or elt2.table is None:
            raise ValueError("Both elements must have a table to perform merging.")

        offset_row = elt1.table.num_rows
        merged_cells = elt1.table.cells + [
            TableCell(
                content=cell.content,
                rows=[r + offset_row for r in cell.rows],
                cols=cell.cols,
                is_header=cell.is_header,
                bbox=cell.bbox,
                properties=cell.properties,
            )
            for cell in elt2.table.cells
        ]

        # Create a new Table object with merged cells
        merged_table = Table(cells=merged_cells, column_headers=elt1.table.column_headers)

        title1 = elt1.data["properties"].get("title", "") or ""
        title2 = elt2.data["properties"].get("title", "") or ""
        merged_title = f"{title1} / {title2}".strip(" / ")
        # Create a new TableElement with the merged table and combined metadata
        new_elt = TableElement(
            title=merged_title if merged_title else None,
            columns=elt1.columns if elt1.columns else elt2.columns,
            rows=elt1.rows + elt2.rows if elt1.rows and elt2.rows else None,
            table=merged_table,
            tokens=elt1.tokens + elt2.tokens if elt1.tokens and elt2.tokens else None,
        )

        # Merge binary representations by concatenation
        if elt1.binary_representation is None or elt2.binary_representation is None:
            new_elt.binary_representation = elt1.binary_representation or elt2.binary_representation
        else:
            new_elt.binary_representation = elt1.binary_representation + elt2.binary_representation
        # Merge text representations by concatenation with a newline
        if elt1.text_representation is None or elt2.text_representation is None:
            new_elt.text_representation = elt1.text_representation or elt2.text_representation
        else:
            new_elt.text_representation = elt1.text_representation + "\n" + elt2.text_representation
        # Merge properties by taking the union of the keys
        properties = new_elt.properties
        for k, v in elt1.properties.items():
            properties[k] = v
            if k == DocumentPropertyTypes.PAGE_NUMBER:
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))
        for k, v in elt2.properties.items():
            if properties.get(k) is None:
                properties[k] = v
            # if a page number exists, add it to the set of page numbers for this new element
            if k == DocumentPropertyTypes.PAGE_NUMBER:
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))

        # TO-DO: Currently bbox points to first table bbox, and other bboxs are removed in
        # this process, potential fix can be to have a list of bboxs, and change label
        # of bbox after first as "table_continuation"
        if elt1.bbox is None or elt2.bbox is None:
            new_elt.bbox = elt1.bbox or elt2.bbox
        else:
            new_elt.bbox = BoundingBox(
                elt1.bbox.x1,
                elt1.bbox.y1,
                elt1.bbox.x2,
                elt1.bbox.y2,
            )
        new_elt.properties = properties

        return new_elt

    def customTableHeaderAdditionFilter(self, elements):

        dic = defaultdict(str)

        # First pass: capture headers
        for ele in elements:
            if ele.type in ["table", "Image", "Formula"]:
                continue
            elif ele.type in ["Text", "Title", "Page-header", "Section-header", "Caption"]:
                if ele.text_representation is not None:
                    text_rep = ele.text_representation.strip()
                if text_rep == "":
                    continue
                if re.search(self.regex_pattern, text_rep):
                    dic[ele["properties"]["page_number"]] = text_rep + " "

        # Second pass: update table elements with headers, done in separate loops since
        # table headers can be within table elements as well or after them
        for ele in elements:
            if ele.type == "table" and isinstance(ele["table"], Table):
                ele.text_representation = dic[ele["properties"]["page_number"]] + ele.text_representation
                if ele["properties"]["title"]:
                    ele["properties"]["title"] = (
                        ele["properties"]["title"] + "\n" + dic[ele["properties"]["page_number"]]
                    )
                else:
                    ele["properties"]["title"] = dic[ele["properties"]["page_number"]]
        return elements

    def process_llm_query(self, document):
        # TO-DO: Add async llm query
        llm_query_agent = LLMTextQueryAgent(prompt=self.llm_prompt, element_type="table", llm=self.llm, table_cont=True)
        llm_results = llm_query_agent.execute_query(document)
        return llm_results


class HeaderAugmenterMerger(ElementMerger):
    """
    The ``HeaderAugmenterMerger`` groups together different elements in a Document and enhances the text
    representation of the elements by adding the preceeding section-header/title.

    - It merges certain elements ("Text", "List-item", "Caption", "Footnote", "Formula", "Page-footer", "Page-header").
    - It merges consecutive ("Section-header", "Title") elements.
    - It adds the preceeding section-header/title to the text representation of the elements (including tables/images).
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

        for element in document.elements:
            if element.type in ["Section-header", "Title"]:
                element.data["_header"] = element.text_representation

        to_merge = [self.preprocess_element(e) for e in document.elements]
        new_elements = [to_merge[0]]
        for element in to_merge[1:]:
            if self.should_merge(new_elements[-1], element):
                new_elements[-1] = self.merge(new_elements[-1], element)
            else:
                new_elements.append(element)
        document.elements = [
            self.postprocess_element(e) for e in new_elements if e.type not in ["Section-header", "Title"]
        ]
        return document

    def should_merge(self, element1: Element, element2: Element) -> bool:
        # deal with empty elements
        if (
            DocumentPropertyTypes.PAGE_NUMBER not in element1.properties
            or DocumentPropertyTypes.PAGE_NUMBER not in element2.properties
            or element1.type is None
            or element2.type is None
        ):
            return False

        # Conditionally prevent merging across pages
        if (
            not self.merge_across_pages
            and element1.properties[DocumentPropertyTypes.PAGE_NUMBER]
            != element2.properties[DocumentPropertyTypes.PAGE_NUMBER]
        ):
            return False

        if element1.data["token_count"] + 1 + element2.data["token_count"] > self.max_tokens and element2.type not in [
            "Section-header",
            "Title",
        ]:
            # Add header to next element
            element2["_header"] = element1.get("_header")
            if element1.get("_header"):
                if element2.text_representation:
                    element2.text_representation = element1["_header"] + "\n" + element2.text_representation
                else:
                    element2.text_representation = element1["_header"]
            return False

        # Merge consecutive section headers/titles and save as a section-header element
        if element1.type in ["Section-header", "Title"] and element2.type in ["Section-header", "Title"]:
            return True

        # MERGE adjacent 'text' elements
        text_like = {"Text", "List-item", "Caption", "Footnote", "Formula", "Page-footer", "Page-header", "Section"}
        if (
            (element1 is not None)
            and (element1.type in text_like)
            and (element2 is not None)
            and (element2.type in text_like)
        ):
            return True

        # Add header to next element (images, tables)
        if element2.type not in ["Section-header", "Title"]:
            element2.data["_header"] = element1.get("_header")
            if element2.text_representation:
                if element2.data["_header"]:
                    element2.text_representation = element2.data["_header"] + "\n" + element2.text_representation
            else:
                element2.text_representation = element2.data["_header"]
        return False

    def merge(self, elt1: Element, elt2: Element) -> Element:
        """Merge two elements; the new element's fields will be set as:
            - type: "Section-header", "Text"
            - binary_representation: elt1.binary_representation + elt2.binary_representation
            - text_representation: elt1.text_representation + elt2.text_representation
            - bbox: the minimal bbox that contains both elt1's and elt2's bboxes
            - properties: elt1's properties + any of elt2's properties that are not in elt1
            note: if elt1 and elt2 have different values for the same property, we take elt1's value
            note: if any input field is None we take the other element's field without merge logic

        Args:
            element1 (Element): the first element (numbers of tokens in it is stored by `preprocess_element`
                                as element1["token_count"])
            element2 (Element): the second element (numbers of tokens in it is stored by `preprocess_element`
                                as element2["token_count"])

        Returns:
            Element: a new merged element from the inputs (and number of tokens in it)
        """
        tok1 = elt1.data["token_count"]
        tok2 = elt2.data["token_count"]
        new_elt = Element()

        if elt1.type in ["Section-header", "Title"] and elt2.type in ["Section-header", "Title"]:
            new_elt.type = "Section-header"
        else:
            new_elt.type = "Text"

        # Merge binary representations by concatenation
        if elt1.binary_representation is None or elt2.binary_representation is None:
            new_elt.binary_representation = elt1.binary_representation or elt2.binary_representation
        else:
            new_elt.binary_representation = elt1.binary_representation + elt2.binary_representation

        # Merge text representations by concatenation with a newline
        new_elt_text_representation = "\n".join(filter(None, [elt1.text_representation, elt2.text_representation]))
        new_elt.text_representation = new_elt_text_representation if new_elt_text_representation else None
        if elt1.text_representation is None or elt2.text_representation is None:
            new_elt.data["token_count"] = tok1 + tok2
        else:
            new_elt.data["token_count"] = tok1 + 1 + tok2

        # Merge bbox by taking the coords that make the largest box
        if elt1.bbox is None and elt2.bbox is None:
            pass
        elif elt1.bbox is None or elt2.bbox is None:
            new_elt.bbox = elt1.bbox or elt2.bbox
        else:
            # TODO: Make bbox work across pages
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
            if k == DocumentPropertyTypes.PAGE_NUMBER:
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))
        for k, v in elt2.properties.items():
            if properties.get(k) is None:
                properties[k] = v
            # if a page number exists, add it to the set of page numbers for this new element
            if k == DocumentPropertyTypes.PAGE_NUMBER:
                properties["page_numbers"] = properties.get("page_numbers", list())
                properties["page_numbers"] = list(set(properties["page_numbers"] + [v]))
        if elt1.type in ["Section-header", "Title"] and elt2.type in ["Section-header", "Title"]:
            if elt1.get("_header") is None or elt2.get("_header") is None:
                new_elt.data["_header"] = elt1.get("_header") or elt2.get("_header")
            else:
                new_elt.data["_header"] = elt1.data["_header"] + "\n" + elt2.data["_header"]
        else:
            new_elt.data["_header"] = elt1.get("_header")
        new_elt.properties = properties

        return new_elt


class Merge(SingleThreadUser, NonGPUUser, Map):
    """
    Merge Elements into fewer large elements
    """

    def __init__(self, child: Node, merger: ElementMerger, **kwargs):
        super().__init__(child, f=merger.merge_elements, **kwargs)
