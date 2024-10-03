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


def mkTable(s: str, page: int, x: float, y: float) -> TableElement:
    cells: list[TableCell] = []
    idx = 0
    for row in range(4):
        for col in range(4):
            idx += 1
            cell = TableCell(f"{s}{idx}", [row], [col], is_header=(row == 0))
            cells.append(cell)
    return elemFromTable(Table(cells), page, x, y)


def elemFromTable(table: Table, page: int, x: float, y: float) -> TableElement:
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


# These are taken from data/test_table.py...


def test_gap() -> None:
    table = Table(
        [
            TableCell(content="1", rows=[0], cols=[0]),
            TableCell(content="2", rows=[0], cols=[1]),
            TableCell(content="4", rows=[1], cols=[1]),
        ]
    )
    te = elemFromTable(table, 1, 0.1, 0.1)
    s = elements_to_markdown([te])
    answer = """
|  |  |
| ----- | ----- |
| 1 | 2 |
|  | 4 |

"""
    assert s == answer


def test_shenanigans() -> None:
    table = Table(
        [
            TableCell(content="A", rows=[0], cols=[0], is_header=False),
            TableCell(content="B", rows=[0, 1, 2], cols=[1], is_header=False),
            TableCell(content="C", rows=[0], cols=[2], is_header=False),
            TableCell(content="D", rows=[1, 2], cols=[0], is_header=False),
            TableCell(content="E", rows=[1], cols=[2], is_header=False),
            TableCell(content="F", rows=[2], cols=[2], is_header=False),
            TableCell(content="G", rows=[3], cols=[0], is_header=False),
            TableCell(content="H", rows=[3], cols=[1], is_header=False),
            TableCell(content="|", rows=[3], cols=[2], is_header=False),
        ]
    )
    te = elemFromTable(table, 1, 0.1, 0.1)
    s = elements_to_markdown([te])
    answer = """
|  |  |  |
| ----- | ----- | ----- |
| A | B | C |
| D | B | E |
| D | B | F |
| G | H | \\| |

"""
    assert s == answer


def test_multi_rowcol() -> None:
    table = Table(
        [
            TableCell(content="multi head", rows=[0, 1], cols=[0, 1], is_header=True),
            TableCell(content="head2_1", rows=[0], cols=[2], is_header=True),
            TableCell(content="head2_2", rows=[1], cols=[2], is_header=True),
            TableCell(content="1", rows=[2], cols=[0], is_header=False),
            TableCell(content="2", rows=[2], cols=[1], is_header=False),
            TableCell(content="3", rows=[2], cols=[2], is_header=False),
            TableCell(content="4", rows=[3], cols=[0], is_header=False),
            TableCell(content="5", rows=[3], cols=[1], is_header=False),
            TableCell(content="6", rows=[3], cols=[2], is_header=False),
        ]
    )
    te = elemFromTable(table, 1, 0.1, 0.1)
    s = elements_to_markdown([te])
    answer = """
| multi head | multi head | head2_1 head2_2 |
| ----- | ----- | ----- |
| 1 | 2 | 3 |
| 4 | 5 | 6 |

"""
    assert s == answer


def test_smithsonian() -> None:
    table = Table(
        [
            TableCell(content="Grade.", rows=[0, 1], cols=[0], is_header=True),
            TableCell(content="Yield Point.", rows=[0, 1], cols=[1], is_header=True),
            TableCell(content="Ultimate tensile strength", rows=[0], cols=[2, 3], is_header=True),
            TableCell(content="Per cent elong. 50.8 mm or 2 in.", rows=[0, 1], cols=[4], is_header=True),
            TableCell(content="Per cent reduct. area.", rows=[0, 1], cols=[5], is_header=True),
            TableCell(content="kg/mm2", rows=[1], cols=[2], is_header=True),
            TableCell(content="lb/in2", rows=[1], cols=[3], is_header=True),
            TableCell(content="Hard", rows=[2], cols=[0]),
            TableCell(content="0.45 ultimate", rows=[2], cols=[1]),
            TableCell(content="56.2", rows=[2], cols=[2]),
            TableCell(content="80,000", rows=[2], cols=[3]),
            TableCell(content="15", rows=[2], cols=[4]),
            TableCell(content="20", rows=[2], cols=[5]),
            TableCell(content="Medium", rows=[3], cols=[0]),
            TableCell(content="0.45 ultimate", rows=[3], cols=[1]),
            TableCell(content="49.2", rows=[3], cols=[2]),
            TableCell(content="70,000", rows=[3], cols=[3]),
            TableCell(content="18", rows=[3], cols=[4]),
            TableCell(content="25", rows=[3], cols=[5]),
            TableCell(content="Soft", rows=[4], cols=[0]),
            TableCell(content="0.45 ultimate", rows=[4], cols=[1]),
            TableCell(content="42.2", rows=[4], cols=[2]),
            TableCell(content="60,000", rows=[4], cols=[3]),
            TableCell(content="22", rows=[4], cols=[4]),
            TableCell(content="30", rows=[4], cols=[5]),
        ],
        caption="Specification values: Steel, Castings, Ann. A.S.T.M. A27-16, Class B;* P max. 0.06; S max. 0.05.",
    )
    te = elemFromTable(table, 1, 0.1, 0.1)
    s = elements_to_markdown([te])
    answer = """
| Grade. | Yield Point. | Ultimate tensile strength kg/mm2 | Ultimate tensile strength lb/in2 | Per cent elong. 50.8 mm or 2 in. | Per cent reduct. area. |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Hard | 0.45 ultimate | 56.2 | 80,000 | 15 | 20 |
| Medium | 0.45 ultimate | 49.2 | 70,000 | 18 | 25 |
| Soft | 0.45 ultimate | 42.2 | 60,000 | 22 | 30 |

Specification values: Steel, Castings, Ann. A.S.T.M. A27-16, Class B;* P max. 0.06; S max. 0.05.
"""  # noqa
    assert s == answer
