import cydifflib

from sycamore.schema import DataType
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.transforms.property_extraction.strategy import RichProperty

# Higher number here means slower but more accurate. 30 seemed decent.
MAX_FUZZY_FIND_PADDING = 30


def refine_attribution(prop: RichProperty, doc: Document) -> RichProperty:
    if prop.value is None:
        prop.attribution = None
        return prop

    # Traverse the JSON
    if prop.type == DataType.OBJECT:
        d = prop.value
        for k, v in d.items():
            d[k] = refine_attribution(v, doc)
        return prop
    if prop.type == DataType.ARRAY:
        ls = prop.value
        for i, v in enumerate(ls):
            ls[i] = refine_attribution(v, doc)
        return prop

    att = prop.attribution
    if att is None:
        return prop
    if len(att.element_indices) == 0:
        elts_to_check = doc.elements
    else:
        elts_to_check = [doc.elements[i] for i in att.element_indices]

    # Check all elements for an exact match before looking for a fuzzy match
    winner = -1.0, (-1, -1), -1
    for elt in elts_to_check:
        eidx = elt.element_index
        if eidx is None:
            continue
        if (span := find_exact(elt, prop)) is not None:
            assert isinstance(eidx, int), "wtf mypy"
            winner = 1.0, span, eidx
            break

    # If no elements contain an exact match, try fuzzy matching
    if winner[0] == -1:
        for elt in elts_to_check:
            eidx = elt.element_index
            if eidx is None:
                continue
            score, span = find_fuzzy(elt, prop)
            if score > winner[0]:
                assert isinstance(eidx, int), "wtf mypy"
                winner = score, span, eidx
            if score == 1.0:
                break

    # Attach results
    score, (begin, end), idx = winner
    if score > 0:
        att.element_indices = [idx]
        att.page = doc.elements[idx].properties.get("page_number")
        att.bbox = doc.elements[idx].bbox
        att.text_span = begin, end
        att.text_match_score = score
        att.text_snippet = (doc.elements[idx].text_representation or "")[begin:end]
    return prop


def find_exact(element: Element, prop: RichProperty) -> tuple[int, int] | None:
    if element.text_representation is None or prop.value is None:
        return None

    proptext = str(prop.value)
    elttext = element.text_representation

    if (i := elttext.find(proptext)) != -1:
        return (i, i + len(proptext))

    return None


def find_fuzzy(element: Element, prop: RichProperty) -> tuple[float, tuple[int, int]]:
    if element.text_representation is None or prop.value is None:
        return 0.0, (-1, -1)
    proptext = str(prop.value)
    elttext = element.text_representation

    initial_window_size = len(proptext)
    matcher = cydifflib.SequenceMatcher(a=proptext)

    max_r = -1
    max_window = 0, 0
    # First pass: find best match of same size as property text
    for i in range(max(len(elttext) - initial_window_size + 1, 1)):
        matcher.set_seq2(elttext[i : i + initial_window_size])
        if (r := matcher.ratio()) > max_r:
            max_r = r
            max_window = i, i + initial_window_size

    # Second pass: expand the previously found best match to see if we can get better
    start, end = max_window
    max_padding = min(initial_window_size, MAX_FUZZY_FIND_PADDING)
    for padding in range(2, max_padding):
        for i in range(padding):
            cand = elttext[start - i : end - i + padding]
            matcher.set_seq2(cand)
            if (r := matcher.ratio()) > max_r:
                max_r = r
                max_window = start - i, end - i + padding
    return max_r, max_window
