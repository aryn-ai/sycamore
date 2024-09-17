"""
Utilities for converting a list of Elements into Markdown-formatted text.

TODO:
- maybe insert horizontal rules at page breaks
- handle numbered lists
"""

from typing import cast

from sycamore.data import Element, TableElement


SKIP_TYPES = {"page-header", "page-footer", "image"}


def elements_to_markdown(elems: list[Element]) -> str:
    """
    This is the main function of interest.
    Assumes elements are sorted as per bbox_sort.bbox_sort_document().
    """
    label_lists(elems)
    s = ""
    last = [-1, -1, -1, -1]
    for elem in elems:
        type = elem_type(elem).lower()
        if type in SKIP_TYPES:
            continue
        bbox = elem.data.get("bbox")
        if bbox:
            if bbox_eq(bbox, last):
                continue
            last = bbox
        if type == "table":
            s += render_table(cast(TableElement, elem))
            continue
        text = elem_text(elem).strip()
        if not text:
            continue
        text = text.replace("\n", " ")
        if type == "title":
            s += f"\n# {text}\n\n"
        elif type == "section-header":
            s += f"\n## {text}\n\n"
        elif type == "list-item":
            tup = elem.data.get("_listctx")
            if tup:
                indent, bullet = tup
                n = len(indent) + len(bullet)
                t = text[n:]
                if not t[0].isspace():
                    t = " " + t
                s += f"{indent}-{t}\n"
            else:
                s += f"- {text}\n"
        elif type in ("caption", "footnote"):
            s += f"\n{text}\n\n"
        else:
            s += text + "\n"
    return s


def render_table(elem: TableElement) -> str:
    table = elem.table
    if not table:
        return ""
    nrow = table.num_rows
    ncol = table.num_cols
    cells = table.cells
    matrix = [[""] * ncol for _ in range(nrow)]
    hdr_max = -1
    for cell in cells:
        hdr = cell.is_header
        for row in cell.rows:
            if hdr:
                hdr_max = max(hdr_max, row)
            for col in cell.cols:
                if cell.content:
                    matrix[row][col] = cell.content
    sep = "| " + " | ".join(["-----" for _ in range(ncol)]) + " |\n"
    s = "\n"
    if hdr_max < 0:
        s += "|  " * ncol + "|\n"
        s += sep
    for row in range(nrow):
        s += "| " + " | ".join(matrix[row]) + " |\n"
        if row == hdr_max:
            s += sep
    return s + "\n"


def label_lists(elems: list[Element]) -> None:
    """
    Assumes elements are already sorted in reading order.
    """
    num = len(elems)
    idx = 0
    while idx < num:
        if elem_type(elems[idx]) == "list-item":
            end = idx + 1
            while (end < num) and (elem_type(elems[idx]) == "list-item"):
                end += 1
            n = end - idx
            if n > 1:
                indent, bullet = common_prefix(elems, idx, end)
                if indent or bullet:
                    for i in range(idx, end):
                        elems[i].data["_listctx"] = (indent, bullet)
            idx = end
        else:
            idx += 1


def common_prefix(elems: list[Element], beg: int, end: int) -> tuple[str, str]:
    ary = [elem_text(elems[i]) for i in range(beg, end)]
    num = len(ary)
    lng = len(min(ary, key=len))
    indent = ""
    bullet = ""
    for pos in range(0, lng):
        ch = ary[0][pos]
        same = True
        for idx in range(1, num):
            if ch != ary[idx][pos]:
                same = False
                break
        if (not same) or ch.isalnum():
            break
        if ch.isspace():
            if bullet:
                break
            else:
                indent += ch
        else:
            bullet += ch
    return indent, bullet


def bbox_eq(a, b) -> bool:
    for idx in range(4):
        d = abs(b[idx] - a[idx])
        if d >= 0.01:
            return False
    return True


def elem_text(elem: Element) -> str:
    return elem.data.get("text_representation", "")


def elem_type(elem: Element) -> str:
    return elem.data.get("type", "")
