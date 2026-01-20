from abc import abstractmethod, ABC
from typing import Any, Optional
import cydifflib

from sycamore.datatype import DataType
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.transforms.property_extraction.types import AttributionValue, RichProperty

# Higher number here means slower but more accurate. 30 seemed decent.
MAX_FUZZY_FIND_PADDING = 30


class AttributionStrategy(ABC):
    """Class describing how attribution is computed for extracted properties."""

    @abstractmethod
    def prediction_to_rich_property(self, prediction: dict[str, Any]) -> RichProperty:
        """Returns a RichProperty constructed from the given prediction dictionary.

        In many cases this will just be RichProperty.from_prediction, but some
        strategies may encode attribution information in the prediction
        dictionary.
        """
        pass

    @abstractmethod
    def default_attribution(
        self, prop: RichProperty, doc: Document, elements: list[Element]
    ) -> Optional[AttributionValue]:
        """Returns the default attribution for the given RichProperty"""
        pass

    @abstractmethod
    def refine_attribution(self, prop: RichProperty, doc: Document) -> RichProperty:
        """Compute the final attribution given the initial RichProperty and the source Document"""
        pass


class TextMatchAttributionStrategy(AttributionStrategy):
    """Attribution strategy that uses text matching on the returned values."""

    def prediction_to_rich_property(self, prediction: dict[str, Any]) -> RichProperty:
        return RichProperty.from_prediction(prediction)

    def default_attribution(
        self, prop: RichProperty, doc: Document, elements: list[Element]
    ) -> Optional[AttributionValue]:
        return AttributionValue(
            element_indices=[e.element_index if e.element_index is not None else -1 for e in elements]
        )

    def refine_attribution(self, prop: RichProperty, doc: Document) -> RichProperty:
        # TODO: Rewrite with zip_traverse
        if prop.value is None:
            prop.attribution = None
            return prop

        # Traverse the JSON
        if prop.type == DataType.OBJECT:
            prop.attribution = None
            d = prop.value
            for k, v in d.items():
                d[k] = self.refine_attribution(v, doc)
            return prop
        if prop.type == DataType.ARRAY:
            prop.attribution = None
            ls = prop.value
            for i, v in enumerate(ls):
                ls[i] = self.refine_attribution(v, doc)
            return prop

        att = prop.attribution
        if att is None:
            return prop

        eid_map = {e.element_index: i for i, e in enumerate(doc.elements)}
        if len(att.element_indices) == 0:
            elts_to_check = doc.elements
        else:
            elts_to_check = [doc.elements[eid_map[i]] for i in att.element_indices]

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
            att.page = doc.elements[eid_map[idx]].properties.get("page_number")
            att.bbox = doc.elements[eid_map[idx]].bbox
            att.text_span = begin, end
            att.text_match_score = score
            att.text_snippet = (doc.elements[eid_map[idx]].text_representation or "")[begin:end]
        return prop


class LLMAttributionStrategy(AttributionStrategy):
    """Attribution strategy that asks the LLM to provide element indices directly."""

    def __init__(self, page_level_attribution: bool = False):
        self.page_level_attribution = page_level_attribution

    def prediction_to_rich_property(self, prediction: dict[str, Any]) -> RichProperty:
        def _recurse(parent: RichProperty, name: Optional[str], prediction: Any) -> None:

            if isinstance(prediction, list) and len(prediction) == 2 and isinstance(prediction[1], (int, type(None))):
                actual_pred = prediction[0]
                predicted_attribution = prediction[1] if prediction[1] is not None else None
                if self.page_level_attribution:
                    attribution = AttributionValue(element_indices=[], page=predicted_attribution) if predicted_attribution is not None else None
                else:
                    attribution = AttributionValue(element_indices=[prediction[1]]) if prediction[1] is not None else None
            else:
                actual_pred = prediction
                attribution = None

            rp = RichProperty.from_single_property(name, actual_pred)
            rp.attribution = attribution

            if isinstance(actual_pred, dict):
                for k, v in actual_pred.items():
                    _recurse(rp, k, v)
            elif isinstance(actual_pred, list):
                for item in actual_pred:
                    _recurse(rp, None, item)

            parent._add_subprop(rp)

        root = RichProperty(name=None, value={}, type=DataType.OBJECT)
        for k, v in prediction.items():
            _recurse(root, k, v)

        return root

    def default_attribution(
        self, prop: RichProperty, doc: Document, elements: list[Element]
    ) -> Optional[AttributionValue]:
        return None

    def refine_attribution(self, prop: RichProperty, doc: Document) -> RichProperty:
        eid_map = {e.element_index: i for i, e in enumerate(doc.elements)}

        def elt_to_page(idx: tuple[int] | list[int]) -> Optional[int]:
            if idx[0] not in eid_map:
                page = None
            else:
                page = doc.elements[eid_map[idx[0]]].properties.get("page_number")
            return page

        def _recurse(prop: RichProperty) -> RichProperty:
            # TODO: Rewrite with zip_traverse
            if prop.value is None:
                prop.attribution = None
                return prop

            # Traverse the JSON
            if prop.type == DataType.OBJECT:
                prop.attribution = None
                d = prop.value
                source_elems = set()

                for k, v in d.items():
                    d[k] = self.refine_attribution(v, doc)
                    child_attribution = d[k].attribution
                    child_indices = (None,) if not child_attribution else child_attribution.element_indices
                    source_elems.add(tuple(child_indices))

                if len(source_elems) == 1 and (idx := source_elems.pop()) != (None,):
                    prop.attribution = AttributionValue(element_indices=list(idx), page=elt_to_page(idx))
                return prop

            if prop.type == DataType.ARRAY:
                prop.attribution = None
                ls = prop.value

                source_elems = set()

                for i, v in enumerate(ls):
                    ls[i] = self.refine_attribution(v, doc)

                    child_attribution = ls[i].attribution
                    child_indices = (None,) if not child_attribution else child_attribution.element_indices
                    source_elems.add(tuple(child_indices))

                if len(source_elems) == 1 and (idx := source_elems.pop()) != (None,):
                    prop.attribution = AttributionValue(element_indices=list(idx), page=elt_to_page(idx))

                return prop

            att = prop.attribution

            if att is not None:
                att.page = elt_to_page(att.element_indices)

            return prop

        return _recurse(prop)


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
