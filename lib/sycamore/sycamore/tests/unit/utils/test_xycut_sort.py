from typing import Any, Optional

from sycamore.data import Document, Element
from sycamore.utils.xycut import xycut_sort_page
from sycamore.utils.element_sort import sort_elements, sort_document


def mkElem(
    left: float, top: float, right: float, bot: float, page: Optional[int] = None, type: str = "Text"
) -> Element:
    d: dict[str, Any] = {"bbox": (left, top, right, bot), "type": type}
    if page is not None:
        d["properties"] = {"page_number": page}
    return Element(d)


def test_page_basic() -> None:
    # e1, e2 in left  column, e1.top < e2.top
    # e0, e3 in right column, e0.top < e3.top
    # e4 full width, at top
    e0 = mkElem(0.59, 0.25, 0.90, 0.60)
    e1 = mkElem(0.10, 0.26, 0.40, 0.51)
    e2 = mkElem(0.10, 0.58, 0.40, 0.90)
    e3 = mkElem(0.60, 0.65, 0.90, 0.85)
    e4 = mkElem(0.15, 0.10, 0.85, 0.15)
    elems = [e0, e1, e2, e3, e4]
    xycut_sort_page(elems)
    answer = [e4, e1, e2, e0, e3]
    assert elems == answer


def test_elements_basic() -> None:
    # e1.top < e0.top = e2.top, e0.left < e2.left both on left
    e0 = mkElem(0.20, 0.50, 0.45, 0.70, 3)
    e1 = mkElem(0.20, 0.21, 0.45, 0.41, 3)
    e2 = mkElem(0.51, 0.50, 0.90, 0.70, 3)

    # e4, e5 in left column, e4.top < e5.top
    # e3, e6 in right column, e3.top < e6.top
    e3 = mkElem(0.59, 0.25, 0.90, 0.60, 1)
    e4 = mkElem(0.10, 0.26, 0.40, 0.51, 1)
    e5 = mkElem(0.10, 0.58, 0.40, 0.90, 1)
    e6 = mkElem(0.60, 0.65, 0.90, 0.85, 1)

    # all the same, test stable
    e7 = mkElem(0.20, 0.21, 0.90, 0.41, 2)
    e8 = mkElem(0.20, 0.21, 0.90, 0.41, 2)
    e9 = mkElem(0.20, 0.21, 0.90, 0.41, 2)

    elems = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]
    sort_elements(elems, mode="xycut")
    answer = [e4, e5, e3, e6, e7, e8, e9, e1, e0, e2]
    assert elems == answer
    assert_element_index_sorted(elems)


def test_document_basic() -> None:
    e0 = mkElem(0.1, 0.5, 0.9, 0.6, 3)
    e1 = mkElem(0.1, 0.1, 0.9, 0.2, 3)
    e2 = mkElem(0.1, 0.5, 0.9, 0.6, 1)
    e3 = mkElem(0.1, 0.1, 0.9, 0.2, 1)
    e4 = mkElem(0.1, 0.5, 0.9, 0.6, 2)
    e5 = mkElem(0.1, 0.1, 0.9, 0.2, 2)
    doc = Document()
    doc.elements = [e0, e1, e2, e3, e4, e5]
    sort_document(doc, mode=xycut_sort_page)
    answer = [e3, e2, e5, e4, e1, e0]
    assert doc.elements == answer
    assert_element_index_sorted(doc.elements)


def test_page_footer() -> None:
    # e1, e2 in left  column, e1.top < e2.top
    # e0, e3 in right column, e0.top < e3.top
    # e4 full width, at top
    # e5 in left column, but page-footer
    e0 = mkElem(0.59, 0.25, 0.90, 0.60)
    e1 = mkElem(0.10, 0.26, 0.40, 0.51)
    e2 = mkElem(0.10, 0.58, 0.40, 0.90)
    e3 = mkElem(0.60, 0.65, 0.90, 0.85)
    e4 = mkElem(0.15, 0.10, 0.85, 0.15)
    e5 = mkElem(0.25, 0.95, 0.75, 1.0, type="Page-footer")
    elems = [e0, e1, e2, e3, e4, e5]
    xycut_sort_page(elems)
    answer = [e4, e1, e2, e0, e3, e5]
    assert elems == answer


def test_no_cut() -> None:
    e0 = mkElem(0.40, 0.70, 0.90, 0.90)
    e1 = mkElem(0.10, 0.40, 0.30, 0.90)
    e2 = mkElem(0.70, 0.10, 0.90, 0.60)
    e3 = mkElem(0.10, 0.10, 0.60, 0.30)
    elems = [e0, e1, e2, e3]
    xycut_sort_page(elems)
    answer = [e3, e1, e0, e2]  # what bbox_sort gives
    assert elems == answer


# bbox coordinates and reading order from page 9 of
# https://www.aemps.gob.es/medicamentosUsoHumano/informesPublicos/docs/IPT-viekirax-exviera.pdf
g_viekirax_boxes = [
    (0.9159, 0.0231, 0.9825, 0.1116, 0),
    (0.5336, 0.1245, 0.9489, 0.1612, 15),
    (0.0951, 0.1245, 0.5205, 0.1486, 1),
    (0.0945, 0.1524, 0.5202, 0.3006, 2),
    (0.5339, 0.1686, 0.9478, 0.2051, 16),
    (0.5340, 0.2126, 0.9529, 0.2492, 17),
    (0.5335, 0.2565, 0.8968, 0.2808, 18),
    (0.5571, 0.2820, 0.9482, 0.3055, 19),
    (0.0945, 0.3046, 0.5198, 0.3655, 3),
    (0.5336, 0.3129, 0.8991, 0.3371, 20),
    (0.5572, 0.3384, 0.9484, 0.3619, 21),
    (0.5332, 0.3689, 0.8977, 0.3932, 22),
    (0.0945, 0.3693, 0.5180, 0.3938, 4),
    (0.5574, 0.3943, 0.9482, 0.4182, 23),
    (0.0947, 0.3974, 0.5187, 0.4221, 5),
    (0.5325, 0.4255, 0.9324, 0.4497, 24),
    (0.0950, 0.4258, 0.5195, 0.4499, 6),
    (0.5324, 0.4575, 0.7744, 0.4693, 25),
    (0.0946, 0.4709, 0.2071, 0.4842, 7),
    (0.0948, 0.4911, 0.5152, 0.5287, 8),
    (0.0945, 0.5355, 0.5029, 0.5725, 9),
    (0.0946, 0.5793, 0.4987, 0.6289, 10),
    (0.0939, 0.6359, 0.5168, 0.7223, 11),
    (0.0948, 0.7290, 0.5058, 0.7785, 12),
    (0.0947, 0.7851, 0.5160, 0.8472, 13),
    (0.0946, 0.8544, 0.5148, 0.8913, 14),
    (0.5568, 0.4700, 0.9550, 0.4941, 26),
    (0.5334, 0.5016, 0.7799, 0.5134, 27),
    (0.5556, 0.5140, 0.9545, 0.5384, 28),
    (0.5324, 0.5452, 0.8140, 0.5575, 29),
    (0.5566, 0.5581, 0.9507, 0.5820, 30),
    (0.5323, 0.5894, 0.8261, 0.6018, 31),
    (0.5564, 0.6020, 0.9550, 0.6265, 32),
    (0.5330, 0.6336, 0.9540, 0.6951, 33),
    (0.5323, 0.7021, 0.9566, 0.7641, 34),
    (0.5323, 0.7706, 0.9559, 0.8326, 35),
    (0.5321, 0.8392, 0.9508, 0.9140, 36),
    (0.4780, 0.9469, 0.5713, 0.9590, 37),
]


def test_viekirax() -> None:
    elems: list[Element] = []
    for tup in g_viekirax_boxes:
        elem = mkElem(tup[0], tup[1], tup[2], tup[3])
        elem.text_representation = str(tup[4])
        elems.append(elem)
    xycut_sort_page(elems)
    for ii, elem in enumerate(elems):
        s = str(ii)
        assert elem.text_representation == s


def assert_element_index_sorted(elements: list[Element]):
    assert all(
        elements[i].element_index < elements[i + 1].element_index for i in range(len(elements) - 1)  # type: ignore
    )
