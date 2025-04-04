from typing import Optional


from sycamore.data import Document, Element
from sycamore.data.document import DocumentPropertyTypes
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import TimeTrace, timetrace


def validBbox(bbox):
    for idx in range(4):
        val = bbox[idx]
        if (val < 0.0) or (val > 1.0):
            return False
    return True


def getBboxLeftTop(elem: Element):
    bbox = elem.data.get("bbox")
    if bbox is None:
        return (0.0, 0.0)
    else:
        return (bbox[0], bbox[1])


def getPageTopLeft(elem: Element):
    bbox = elem.data.get("bbox")
    if bbox is None:
        return (elem.properties[DocumentPropertyTypes.PAGE_NUMBER], 0.0, 0.0)
    else:
        return (elem.properties[DocumentPropertyTypes.PAGE_NUMBER], bbox[1], bbox[0])


def getRow(elem: Element, elements: list[Element]) -> list[Element]:
    rv = [elem]

    bbox = elem.data.get("bbox")
    if bbox is None:
        return rv
    left = bbox[0]
    top = bbox[1]
    right = bbox[2]
    bottom = bbox[3]
    page = elem.properties[DocumentPropertyTypes.PAGE_NUMBER]

    # !!! assuming elements are sorted by y-values
    n = len(elements)
    beg = 0
    end = n
    idx = 0
    while beg < end:
        mid = beg + ((end - beg) // 2)
        melem = elements[mid]
        mpage = melem.properties[DocumentPropertyTypes.PAGE_NUMBER]
        if mpage < page:
            beg = mid + 1
            idx = mid
        elif mpage > page:
            end = mid
        else:
            mbb = melem.data["bbox"]
            mtop = mbb[1]
            if mtop < top:
                beg = mid + 1
                idx = mid
            elif mtop > top:
                end = mid
            else:
                break

    for idx in range(idx, n):
        ee = elements[idx]
        bb = ee.data["bbox"]
        if bb[1] > bottom:
            break
        if bb[3] < top:
            continue
        if (bb[0] > right) or (bb[2] < left):
            rv.append(ee)

    rv.sort(key=getBboxLeftTop)
    return rv


def partOfTwoCol(elem: Element, xmin, xmax) -> bool:
    cc = elem.data.get("_colCnt")
    if (cc is None) or (cc != 2):
        return False
    bb = elem.data.get("bbox")
    if bb is None:
        return False
    left = bb[0]
    width = bb[2] - left
    pageWidth = xmax - xmin
    halfWidth = pageWidth / 2
    if width > halfWidth:
        return False
    frac = (left - xmin) / pageWidth
    return (frac <= 0.1) or ((frac >= 0.45) and (frac <= 0.6))


###############################################################################


class SortByPageBbox(SingleThreadUser, NonGPUUser, Map):
    """
    SortByPageBbox is a transform to reorder the Elements
    in 'natural order', top to bottom using page_number and bbox.

    Args:
        child: The source Node or component that provides the Elements

    Example:
        .. code-block:: python

            source_node = ...
            sorter = SortByPageBbox(child=source_node)
            dataset = sorter.execute()
    """

    def __init__(self, child: Node, **resource_args):
        super().__init__(child, f=SortByPageBbox.sort_by_page_bbox, **resource_args)

    @staticmethod
    def sort_by_page_bbox(parent: Document) -> Document:
        with TimeTrace("sortPageBbox"):
            parent.elements.sort(key=getPageTopLeft)
            return parent


###############################################################################


class MarkDropHeaderFooter(SingleThreadUser, NonGPUUser, Map):
    """
    MarkDropHeaderFooter is a transform to add the '_drop' data attribute to
    each Element at the top or bottom X fraction of the page.  Requires
    the 'bbox' attribute.

    Args:
        child: The source Node or component that provides the Elements
        top: The fraction of the page to exclude from the top (def 0.05)
        bottom: The fraction of the page to exclude from the bottom (0.05)

    Example:
        .. code-block:: python

            source_node = ...
            marker = MarkDropHeaderFooter(child=source_node, top=0.05)
            dataset = marker.execute()
    """

    def __init__(self, child: Node, top: float = 0.05, bottom: Optional[float] = None, **resource_args):
        super().__init__(child, f=MarkDropHeaderFooter.mark_drop_header_and_footer, args=[top, bottom], **resource_args)

    @staticmethod
    @timetrace("markHeadFoot")
    def mark_drop_header_and_footer(parent: Document, top: float = 0.05, bottom: Optional[float] = None) -> Document:
        if bottom is None:
            bottom = top

        lo = top
        hi = 1.0 - bottom

        for elem in parent.elements:
            bbox = elem.data.get("bbox")
            if (bbox is not None) and ((bbox[1] > hi) or (bbox[3] < lo)):
                elem.data["_drop"] = True  # mark for removal
        return parent


###############################################################################


class MarkBreakByColumn(SingleThreadUser, NonGPUUser, Map):
    """
    MarkBreakByColumn is a transform that marks '_break' where
    two-column layout changes to full-width layout.  Ranges of two-
    column Elements are also re-sorted left to right.  Elements must
    already be sorted top-to-bottom.

    Args:
        child: The source Node or component that provides the Elements

    Example:
        .. code-block:: python

            source_node = ...
            marker = MarkBreakByColumn(child=source_node)
            dataset = marker.execute()
    """

    def __init__(self, child: Node, **resource_args):
        super().__init__(child, f=MarkBreakByColumn.mark_break_by_column, **resource_args)

    @staticmethod
    @timetrace("makeBreakCol")
    def mark_break_by_column(parent: Document) -> Document:
        elements = parent.elements

        # measure width in-use
        xmin = 1.0  # FIXME are these global?
        xmax = 0.0
        for elem in elements:
            bbox = elem.data.get("bbox")
            if (bbox is not None) and validBbox(bbox):
                xmin = min(xmin, bbox[0])
                xmax = max(xmax, bbox[2])
        if xmin < xmax:
            fullWidth = (xmax - xmin) * 0.8  # fudge
        else:
            fullWidth = 0.8

        # tag elements by column
        for elem in elements:
            if elem.data.get("_colIdx") is None:
                row = getRow(elem, elements)
                if len(row) == 1:
                    bbox = elem.data.get("bbox")
                    if bbox is None:
                        width = 0.0
                    else:
                        width = bbox[2] - bbox[0]
                    if width > fullWidth:
                        cnt = 0  # signal full-width
                    else:
                        cnt = 1
                    elem.data["_colIdx"] = 0
                    elem.data["_colCnt"] = cnt
                else:
                    idx = -1
                    last = 0.0
                    for ee in row:
                        bbox = ee.data["bbox"]
                        if bbox[0] >= last:  # may be stacked vertically
                            idx += 1
                        last = bbox[2]
                        if ee.data.get("_colIdx") is None:
                            ee.data["_colIdx"] = idx
                    for ee in row:
                        if ee.data.get("_colCnt") is None:
                            ee.data["_colCnt"] = idx + 1

        # re-sort ranges of two-column text
        last = 0
        ranges = []
        for idx, elem in enumerate(elements):
            if not partOfTwoCol(elem, xmin, xmax):
                if (idx - last) > 4:
                    ranges.append((last + 1, idx))
                last = idx
        for xx, yy in ranges:
            elements[xx:yy] = sorted(elements[xx:yy], key=getBboxLeftTop)

        # mark breaks due to column transitions
        lastCols = 0
        for elem in elements:
            ecols = elem.data["_colCnt"]
            if ecols != lastCols:
                if ecols == 0:
                    elem.data["_break"] = True
                lastCols = ecols

        return parent
