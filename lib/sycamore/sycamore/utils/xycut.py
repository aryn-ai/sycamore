"""
Sort Elements into reading order based on bboxes using X-Y Cut.

Most useful functions are at the bottom of the file.
"""

from io import StringIO
from typing import Generator, Optional

from sycamore.data import Document, Element
from sycamore.utils.bbox_sort import collect_pages

ElemList = list[Element]
BeginEndList = list[tuple[float, int, Element]]

XAXIS = 0
YAXIS = 1

OPEN = 1
CLOSE = 0  # sorts first


class NodeBase:
    """Like a B-tree of Elements."""

    def __str__(self) -> str:
        sio = StringIO()
        self.to_str_impl("", sio)
        return sio.getvalue().rstrip()

    def to_text(self) -> str:
        sio = StringIO()
        self.to_text_impl(sio)
        return sio.getvalue().rstrip()

    def to_elems(self) -> ElemList:
        elems: ElemList = []
        self.to_elems_impl(elems)
        return elems

    def to_str_impl(self, pfx: str, sio: StringIO) -> None:
        raise NotImplementedError()

    def to_text_impl(self, sio: StringIO) -> None:
        raise NotImplementedError()

    def to_elems_impl(self, elems: ElemList) -> None:
        raise NotImplementedError()


class NodeInner(NodeBase):
    """Node whose chidren are all Nodes."""

    nodes: list[NodeBase]

    def __init__(self) -> None:
        super().__init__()
        self.nodes = []

    def append(self, node: NodeBase) -> "NodeInner":
        self.nodes.append(node)
        return self

    def to_str_impl(self, pfx: str, sio: StringIO) -> None:
        for idx, node in enumerate(self.nodes):
            node.to_str_impl(f"{pfx}{idx}.", sio)

    def to_text_impl(self, sio: StringIO) -> None:
        for node in self.nodes:
            node.to_text_impl(sio)

    def to_elems_impl(self, elems: ElemList) -> None:
        for node in self.nodes:
            node.to_elems_impl(elems)


class NodeLeaf(NodeBase):
    """Node whose children are all lists of elements."""

    elists: list[ElemList]

    def __init__(self) -> None:
        super().__init__()
        self.elists = [[]]

    def append(self, elem: Element) -> "NodeLeaf":
        self.elists[-1].append(elem)
        return self

    def extend(self, elist: ElemList) -> "NodeLeaf":
        self.elists[-1].extend(elist)
        return self

    def advance(self) -> "NodeLeaf":
        self.elists.append([])
        return self

    def finalize(self) -> "NodeLeaf":
        if not self.elists[-1]:
            self.elists.pop()
        return self

    def cansplit(self) -> bool:
        try:
            elem = self.elists[0][0]
            next = self.elists[1]  # noqa: F841
            return isinstance(elem, Element)
        except (IndexError, TypeError):
            return False

    def to_str_impl(self, pfx: str, sio: StringIO) -> None:
        for idx, elist in enumerate(self.elists):
            for i, elem in enumerate(elist):
                s = elem_to_str(elem)
                sio.write(f"{pfx}{i}: {s}\n")

    def to_text_impl(self, sio: StringIO) -> None:
        for elist in self.elists:
            for elem in elist:
                if tr := elem.text_representation:
                    sio.write(tr)
                    sio.write("\n\n")

    def to_elems_impl(self, elems: ElemList) -> None:
        for elist in self.elists:
            elems.extend(elist)


###############################################################################


def get_bbox(elem: Element) -> tuple[float, float, float, float]:
    if bbox := elem.data.get("bbox"):
        return bbox
    return (1.0, 1.0, 1.0, 1.0)


def bbox_to_str(bbox: tuple[float, float, float, float]) -> str:
    a, b, c, d = [int(1000 * x) for x in bbox]
    return f"({a:03d},{b:03d})({c:03d},{d:03d})"


def elem_to_str(elem: Element) -> str:
    s = bbox_to_str(get_bbox(elem))
    if tr := elem.text_representation:
        nonl = tr.replace("\n", " ")[:32]
        s += f"[{nonl}]"
    return s


def make_begin_end(elems: ElemList, axis: int) -> BeginEndList:
    """Returns array of (coord, isopen, elem)"""
    ary: BeginEndList = []
    for elem in elems:
        bbox = get_bbox(elem)
        aa = bbox[axis]
        bb = bbox[axis + 2]
        if bb < aa:
            aa, bb = bb, aa
        ary.append((aa, OPEN, elem))
        ary.append((bb, CLOSE, elem))
    ary.sort()
    return ary


def gen_overlap(ary: BeginEndList) -> Generator[tuple, None, None]:
    """Yields tuple (coord, isopen, elem, count, width)"""
    count = 0
    cur = ary[0]
    n = len(ary)
    for ii in range(n):
        width = -1.0
        if cur[1] == OPEN:
            count += 1
            next = ary[ii + 1]
        else:
            count -= 1
            if (ii + 1) < n:
                next = ary[ii + 1]
                if count == 0:
                    width = next[0] - cur[0]
        yield (*cur, count, width)
        cur = next


def widest_cut(order: BeginEndList) -> tuple[float, Element]:
    rv = (-1.0, order[0][2])
    if len(order) < 3:  # order is twice as long as elems
        return rv
    for _, _, elem, count, width in gen_overlap(order):
        if count == 0:
            rv = max(rv, (width, elem))
    return rv


def choose_axis(elems: ElemList) -> Optional[tuple[BeginEndList, Element]]:
    xorder = make_begin_end(elems, XAXIS)
    yorder = make_begin_end(elems, YAXIS)
    xw, xe = widest_cut(xorder)
    yw, ye = widest_cut(yorder)
    if max(xw, yw) < 0.0:
        return None
    if xw < yw:
        # yval = get_bbox(ye)[3]
        # s = elem_to_str(ye)
        # print(f"Horiz cut y={yval:.3f} {s}")
        return (yorder, ye)
    else:
        # xval = get_bbox(xe)[2]
        # s = elem_to_str(xe)
        # print(f"Vert cut x={xval:.3f} {s}")
        return (xorder, xe)


def cleave_elems(elems: ElemList) -> NodeLeaf:
    """Binary split across widest gap."""
    node = NodeLeaf()
    if len(elems) < 2:
        node.extend(elems)
    elif (choice := choose_axis(elems)) is None:
        node.extend(elems)
    else:
        order, cut_after = choice
        for _, isopen, elem, cnt, width in gen_overlap(order):
            if not isopen:
                node.append(elem)
                if elem == cut_after:
                    node.advance()
    return node.finalize()


def divide_node(node: NodeLeaf) -> NodeInner:
    """Return replacement Node.  Assume input is leaf."""
    inner = NodeInner()
    for elist in node.elists:
        subnode = cleave_elems(elist)
        # print("....................")
        # print(subnode)
        # print("^^^^^^^^^^^^^^^^^^^^")
        if subnode.cansplit():
            inner.append(divide_node(subnode))  # recursive step
        else:
            inner.append(subnode)
    return inner


def xycut_sorted_page(elems: ElemList) -> ElemList:
    if len(elems) < 2:
        return elems
    flat = NodeLeaf().extend(elems)
    # print("VVVVVVVVVVVVVVVVVVVV")
    # print(flat)
    # print("AAAAAAAAAAAAAAAAAAAA")
    tree = divide_node(flat)
    return tree.to_elems()


def xycut_sorted_elements(elements: ElemList, update_indices: bool = True) -> ElemList:
    flat: ElemList = []
    pages = collect_pages(elements)
    for page in pages:
        elems = xycut_sorted_page(page)
        flat.extend(elems)
    if update_indices:
        for idx, elem in enumerate(flat):
            elem.element_index = idx
    return flat


def xycut_sort_document(doc: Document, update_indices: bool = True) -> None:
    doc.elements = xycut_sorted_elements(doc.elements, update_indices)
