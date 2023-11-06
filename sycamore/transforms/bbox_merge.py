from typing import Any, Dict
from sycamore.data import Document, Element
from sycamore.functions.tokenizer import Tokenizer
from sycamore.transforms.merge_elements import ElementMerger

# TODO:
# - overlap?
# - 3 columns, N columns


def validBbox(bbox):
    for idx in range(4):
        val = bbox[idx]
        if (val < 0.0) or (val > 1.0):
            return False
    return True


def getBboxTop(elem: Element):
    return elem.data["bbox"][1]


def getBboxLeft(elem: Element):
    return elem.data["bbox"][0]


def getBboxLeftTop(elem: Element):
    bb = elem.data["bbox"]
    return (bb[0], bb[1])


def getPageTopLeft(elem: Element):
    bb = elem.data["bbox"]
    return (elem.properties["page_number"], bb[1], bb[0])


def getRow(elem: Element, elements: list[Element]) -> list[Element]:
    page = elem.properties["page_number"]
    bbox = elem.data["bbox"]
    left = bbox[0]
    top = bbox[1]
    right = bbox[2]
    bottom = bbox[3]

    # !!! assuming elements are sorted by y-values
    n = len(elements)
    beg = 0
    end = n
    idx = 0
    while beg < end:
        mid = beg + ((end - beg) // 2)
        melem = elements[mid]
        mpage = melem.properties["page_number"]
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

    rv = [elem]
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


class BboxMerger(ElementMerger):
    def __init__(self, tokenizer: Tokenizer, maxToks: int = 512):
        self.tokenizer = tokenizer
        self.maxToks = maxToks

    def should_merge(self, element1: Element, element2: Element) -> bool:
        return False

    def merge(self, element1: Element, element2: Element) -> Element:
        return element1

    def preprocess_element(self, elem: Element) -> Element:
        if elem.text_representation:
            n = len(self.tokenizer.tokenize(elem.text_representation))
        else:
            n = 0
        elem.data["_tokCnt"] = n
        return elem

    def postprocess_element(self, elem: Element) -> Element:
        del elem.data["token_count"]
        return elem

    def merge_elements(self, document: Document) -> Document:
        if len(document.elements) < 2:
            return document
        elements = document.elements

        # FIXME: make separate transforms to mark/remove these
        for elem in elements:
            tr = elem.text_representation or ""
            if len(tr) <= 1:  # specks of lint
                elem.data["_drop"] = True
        for elem in elements:
            bbox = elem.data["bbox"]  # must be defined for this merger
            if (bbox[1] > 0.95) or (bbox[3] < 0.05):  # headers and footers
                elem.data["_drop"] = True

        for elem in elements:
            self.preprocess_element(elem)

        # measure width in-use
        xmin = 1.0  # FIXME are these global?
        xmax = 0.0
        for elem in elements:
            bbox = elem.data["bbox"]  # make sure it's defined
            if validBbox(bbox):  # avoid bug
                xmin = min(xmin, bbox[0])
                xmax = max(xmax, bbox[2])
        fullWidth = (xmax - xmin) * 0.8  # fudge

        # sort entire document top-to-bottom
        elements.sort(key=getPageTopLeft)

        # mark break at page breaks
        lastPage = elements[0].properties["page_number"]
        for elem in elements:
            page = elem.properties["page_number"]
            if page != lastPage:
                elem.data["_break"] = True
                lastPage = page

        # tag elements by column
        for elem in elements:
            if elem.data.get("_colIdx") is None:
                row = getRow(elem, elements)
                if len(row) == 1:
                    bbox = elem.data["bbox"]
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

        # mark breaks to keep within token limit
        # FIXME: do this in a balanced way
        toks = 0
        for elem in elements:
            etoks = elem.data["_tokCnt"]
            if elem.data.get("_break") or ((toks + etoks) > self.maxToks):
                elem.data["_break"] = True
                toks = 0
            toks += etoks

        # merge elements, honoring marked breaks and drops
        merged = []
        bin = b""
        text = ""
        props: Dict[str, Any] = {}
        bbox = None

        for elem in elements:
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
                if k not in props:  # FIXME: order may matter here
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
