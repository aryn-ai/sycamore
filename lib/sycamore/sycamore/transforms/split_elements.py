from typing import Optional
import logging
from sycamore.data import Document, Element, TableElement
from sycamore.functions.tokenizer import Tokenizer
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace

logger = logging.getLogger(__name__)


class SplitElements(SingleThreadUser, NonGPUUser, Map):
    """
    The SplitElements transform recursively divides elements such that no
    Element exceeds a maximum number of tokens.

    Args:
        child: The source node or component that provides the elements to be split
        tokenizer: The tokenizer to use in counting tokens, should match embedder
        maximum: Maximum tokens allowed in any Element

    Example:
        .. code-block:: python

            node = ...  # Define a source node or component that provides hierarchical documents.
            xform = SplitElements(child=node, tokenizer=tokenizer, 512)
            dataset = xform.execute()
    """

    def __init__(self, child: Node, tokenizer: Tokenizer, maximum: int, **kwargs):
        super().__init__(child, f=SplitElements.split_doc, args=[tokenizer, maximum], **kwargs)

    @staticmethod
    @timetrace("splitElem")
    def split_doc(parent: Document, tokenizer: Tokenizer, max: int) -> Document:
        result = []
        for elem in parent.elements:
            # Ensure the _header does not take up more than a third of the tokens
            # Also avoid max resursive depth error
            if elem.get("_header") and len(tokenizer.tokenize(elem["_header"])) / max > 0.33:
                logger.warning(f"Token limit exceeded, dropping _header: {elem['_header']}")
                del elem["_header"]
            result.extend(SplitElements.split_one(elem, tokenizer, max))
        parent.elements = result
        return parent

    @staticmethod
    def split_one(elem: Element, tokenizer: Tokenizer, max: int, depth: int = 0) -> list[Element]:
        if depth > 20:
            logger.warning(f"Max split depth exceeded, truncating the splitting")
            return [elem]

        txt = elem.text_representation
        if not txt:
            return [elem]
        num = len(tokenizer.tokenize(txt))
        if num <= max:
            return [elem]

        half = len(txt) // 2
        left = half
        right = half + 1

        # FIXME: The table object in the split elements would have the whole table structure rather than split
        newlineFound = False
        if elem.type == "table":
            for jj in range(half // 2):
                if txt[left] == "\n":
                    idx = left + 1
                    newlineFound = True
                    break
                elif txt[right] == "\n":
                    idx = right + 1
                    newlineFound = True
                    break
                left -= 1
                right += 1

        # FIXME: make this work with asian languages
        if not newlineFound:
            left = half
            right = half + 1
            predicates = [  # in precedence order
                lambda c: c in ".!?",
                lambda c: c == ";",
                lambda c: c in "()",
                lambda c: c == ":",
                lambda c: c == ",",
                str.isspace,
            ]
            results: list[Optional[int]] = [None] * len(predicates)

            for jj in range(half // 2):  # stay near middle; avoid the ends
                lchar = txt[left]
                rchar = txt[right]

                go = True
                for ii, predicate in enumerate(predicates):
                    if predicate(lchar):
                        if results[ii] is None:
                            results[ii] = left
                        go = ii != 0
                        break
                    elif predicate(rchar):
                        if results[ii] is None:
                            results[ii] = right
                        go = ii != 0
                        break
                if not go:
                    break

                left -= 1
                right += 1

            idx = half + 1
            for res in results:
                if res is not None:
                    idx = res + 1
                    break

        one = txt[:idx]
        two = txt[idx:]

        ment = elem.copy()
        elem.text_representation = one
        elem.binary_representation = bytes(one, "utf-8")
        if elem.type == "table" and isinstance(elem, TableElement) and elem.table is not None:
            if elem.table.column_headers:
                two = ", ".join(elem.table.column_headers) + "\n" + two
            if elem.data["properties"].get("title"):
                two = elem.data["properties"].get("title") + "\n" + two
        if elem.get("_header"):
            ment.text_representation = ment["_header"] + "\n" + two
        else:
            ment.text_representation = two
        ment.binary_representation = bytes(two, "utf-8")
        aa = SplitElements.split_one(elem, tokenizer, max, depth + 1)
        bb = SplitElements.split_one(ment, tokenizer, max, depth + 1)
        aa.extend(bb)
        return aa
