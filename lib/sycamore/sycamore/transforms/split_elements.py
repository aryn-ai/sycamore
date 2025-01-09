import math
from typing import Optional
import logging
from sycamore.data import Document, Element, TableElement, TableCell, Table, BoundingBox
from sycamore.functions.tokenizer import Tokenizer
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace

logger = logging.getLogger(__name__)

RECURSIVE_SPLIT_MAX_DEPTH = 20


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
        if depth > RECURSIVE_SPLIT_MAX_DEPTH:
            logger.warning("Max split depth exceeded, truncating the splitting")
            return [elem]

        if elem.type == "table" and isinstance(elem, TableElement) and elem.table is not None:
            return SplitElements.split_one_table(elem, tokenizer, max, depth)

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

    @staticmethod
    def split_one_table(element: TableElement, tokenizer: Tokenizer, max_tokens: int, depth: int = 0) -> list[Element]:
        """
        Special handling for tables: If the column header is too big, no amount of splitting the
        rows will save us, as we want to attach the col header to each subtable. In this case,
        split the table horizontally. If the column header is small enough, we can guess the number
        of rows per subtable and try to break the table into chunks of that size for evenness (still
        breaking when we run out of tokens). Special care is taken to adjust the bounding box appropriately.
        """
        if depth > RECURSIVE_SPLIT_MAX_DEPTH:
            logger.warning("Max split depth exceeded, truncating the splitting")
            return [element]

        assert element.table is not None, "Cannot split a table without table structure"

        col_header_len = len(tokenizer.tokenize(", ".join(element.table.column_headers)))
        data_table = element.table.data_cells()
        data_row_lens = [
            len(tokenizer.tokenize(", ".join([c.content for c in data_table.cells if i in c.rows])))
            for i in range(data_table.num_rows)
        ]
        if col_header_len > max_tokens - max(data_row_lens):
            # If there is a row and column header that cannot combine,
            # split table horizontally in half and recurse. Splitting
            # is done by column number rather than text.
            ncols = element.table.num_cols
            if ncols <= 1:
                # One-column table that's too big - turn it into text and split in the traditional way.
                new_elt = element.copy()
                new_elt.data["text_representation"] = new_elt.text_representation
                new_elt.type = "Text"
                return SplitElements.split_one(new_elt, tokenizer, max_tokens, depth + 1)
            # Split the table by splitting the cells into groups.
            elem_cells = [
                TableCell(c.content, c.rows, [cl for cl in c.cols if cl < ncols // 2], c.is_header, c.bbox)
                for c in element.table.cells
                if min(c.cols) < ncols // 2
            ]
            ment_cells = [
                TableCell(
                    c.content, c.rows, [cl - ncols // 2 for cl in c.cols if cl >= ncols // 2], c.is_header, c.bbox
                )
                for c in element.table.cells
                if max(c.cols) >= ncols // 2
            ]
            elem = element.copy()
            ment = element.copy()
            elem.table = Table(elem_cells, element.table.caption)
            ment.table = Table(ment_cells, element.table.caption)
            _reset_table_bbox(elem)
            _reset_table_bbox(ment)
            return SplitElements.split_one_table(
                elem, tokenizer, max_tokens, depth + 1
            ) + SplitElements.split_one_table(ment, tokenizer, max_tokens, depth + 1)
        # We can attach the column header to every row, so break the rows up
        # evenly into groups and form new tables each containing a set of rows.
        # Ensure that each resulting table is below the token limit too.
        data_max_tokens = max_tokens - col_header_len
        header_cells = element.table.header_cells()
        if len(header_cells) > 0:
            n_header_rows = max(r for c in header_cells for r in c.rows) + 1
        else:
            n_header_rows = 0
        # Try to be slightly less greedy by giving each chunk a row limit
        expected_chunks = math.ceil(sum(data_row_lens) / data_max_tokens)
        expected_rows_per_chunk = math.ceil(len(data_row_lens) / expected_chunks)
        curr_len = 0
        curr_rows: list[int] = []
        subtables = []
        for i, drl in enumerate(data_row_lens):
            if curr_len + drl < data_max_tokens and len(curr_rows) < expected_rows_per_chunk:
                curr_rows.append(i)
                curr_len += drl
            else:
                begin, end = curr_rows[0], curr_rows[-1]
                new_table_cells = header_cells + [
                    TableCell(
                        c.content,
                        [r - begin + n_header_rows for r in c.rows if begin <= r <= end],
                        c.cols,
                        c.is_header,
                        c.bbox,
                    )
                    for c in data_table.cells
                    if any(begin <= r <= end for r in c.rows)
                ]
                subtables.append(Table(new_table_cells, element.table.caption))
                curr_len = drl
                curr_rows = [i]

        begin, end = curr_rows[0], curr_rows[-1]
        new_table_cells = header_cells + [
            TableCell(
                c.content, [r - begin + n_header_rows for r in c.rows if begin <= r <= end], c.cols, c.is_header, c.bbox
            )
            for c in data_table.cells
            if any(begin <= r <= end for r in c.rows)
        ]
        subtables.append(Table(new_table_cells, element.table.caption))
        elms = [element.copy() for _ in subtables]
        first = True
        for elm, sbt in zip(elms, subtables):
            elm.table = sbt
            _reset_table_bbox(elm, ignore_header=not first)
            first = False
        return elms


def _reset_table_bbox(te: TableElement, ignore_header: bool = False):
    """
    Set a table element's overall bbox to something that aligns with
    its cells. Specifically, take the median left, right, top, and bottom
    edges of all cells on the corresponding edge of the table. We want
    the median here rather then the extreme because in split tables with
    spanning cells the extreme may be the edge of the spanning cell, which
    may hang outside of the columns or rows of the table we're working with.
    """
    assert te.table is not None
    if ignore_header:
        dc = te.table.data_cells().cells
    else:
        dc = te.table.cells
    if te.bbox is None or all(c.bbox is None for c in dc):
        return
    max_row = max(c.rows[-1] for c in dc)
    max_col = max(c.cols[-1] for c in dc)
    min_row = min(c.rows[0] for c in dc)
    min_col = min(c.cols[0] for c in dc)
    x1s = []
    x2s = []
    y1s = []
    y2s = []
    for c in dc:
        if c.bbox is None:
            continue
        if c.cols[0] == min_col:
            x1s.append(c.bbox.x1)
        if c.cols[-1] == max_col:
            x2s.append(c.bbox.x2)
        if c.rows[0] == min_row:
            y1s.append(c.bbox.y1)
        if c.rows[-1] == max_row:
            y2s.append(c.bbox.y2)
    new_bb = BoundingBox(
        _median(x1s),
        _median(y1s),
        _median(x2s),
        _median(y2s),
    )
    if new_bb is not None:
        te.bbox = new_bb


def _median(nums: list[float]) -> float:
    nums.sort()
    return nums[len(nums) // 2]
