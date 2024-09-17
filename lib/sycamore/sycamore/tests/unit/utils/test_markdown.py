from sycamore.data import Element, TableElement
from sycamore.data.table import Table, TableCell
from sycamore.utils.markdown import elements_to_markdown


def mkElem(type, text, page, x, y, w, h=0.09):
    return Element(
        {
            "type": type,
            "text_representation": text,
            "bbox": (x, y, x + w, y + h),
            "properties": {
                "page_number": page,
            },
        }
    )


def mkText(s: str, page: int, x: float, y: float, w=0.4) -> Element:
    return mkElem("Text", s, page, x, y, w)


def mkTitle(s: str, page: int, x: float, y: float, w=0.4) -> Element:
    return mkElem("Title", s, page, x, y, w)


def mkHead(s: str, page: int, x: float, y: float, w=0.4) -> Element:
    return mkElem("Section-header", s, page, x, y, w)


def mkItem(s: str, page: int, x: float, y: float, w=0.4) -> Element:
    return mkElem("List-item", s, page, x, y, w)


def mkTable(s: str, page: int, x: float, y: float) -> Element:
    cells: list[TableCell] = []
    idx = 0
    for row in range(4):
        for col in range(4):
            idx += 1
            cell = TableCell(f"{s}{idx}", [row], [col], is_header=(row == 0))
            cells.append(cell)
    table = Table(cells)
    elem = Element(
        {
            "type": "Table",
            "bbox": (x, y, x + 0.7, y + 0.19),
            "properties": {
                "page_number": page,
            },
        }
    )
    return TableElement(elem, table=table)


def test_basic() -> None:
    elems = [
        mkTitle("Title", 1, 0.1, 0.1, 0.8),
        mkText("left", 1, 0.1, 0.2),
        mkText("right", 1, 0.6, 0.2),
        mkText("full", 1, 0.1, 0.3, 0.8),
        mkHead("Section", 1, 0.1, 0.4),
        mkItem("Item1", 1, 0.15, 0.5),
        mkItem("Item2", 1, 0.55, 0.5),
        mkItem("Item3", 1, 0.15, 0.6),
        mkItem("Item4", 1, 0.55, 0.6),
        mkTable("Cell", 1, 0.15, 0.7),
    ]
    s = elements_to_markdown(elems)
    answer = """
# Title

left
right
full

## Section

- Item1
- Item2
- Item3
- Item4

| Cell1 | Cell2 | Cell3 | Cell4 |
| ----- | ----- | ----- | ----- |
| Cell5 | Cell6 | Cell7 | Cell8 |
| Cell9 | Cell10 | Cell11 | Cell12 |
| Cell13 | Cell14 | Cell15 | Cell16 |

"""
    assert s == answer
