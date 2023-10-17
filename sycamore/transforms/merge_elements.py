from abc import ABC, abstractmethod

from ray.data import Dataset, ActorPoolStrategy

from sycamore.data import Document, Element, BoundingBox
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Transform, Node
from sycamore.transforms.map import generate_map_class_from_callable
from sycamore.functions.tokenizer import Tokenizer

from typing import Tuple


class ElementMerger(ABC):
    @abstractmethod
    def merge_elements(self, document: Document) -> Document:
        pass


class GreedyTextElementMerger(ElementMerger):
    def __init__(self, tokenizer: Tokenizer, max_tokens: int):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def _should_merge(self, element1: Tuple[Element, int], element2: Tuple[Element, int]) -> bool:
        if element2[0].type == "Title":
            return False
        if element1[1] + 1 + element2[1] > self.max_tokens:
            return False
        return True

    def _merge(self, element1: Tuple[Element, int], element2: Tuple[Element, int]) -> Tuple[Element, int]:
        """
        Merge two elements; the new element's fields will be set as:

            type: "Section"
            binary_representation: elt1.binary_representation + elt2.binary_representation
            text_representation: elt1.text_representation + '\n' + elt2.text_representation
            bbox: the minimal bbox that contains both elt1's and elt2's bboxes
            properties: elt1's properties + any of elt2's properties that are not in elt1
                > note: if elt1 and elt2 have different values for the same property, we take elt1's value

            > note: if any input field is None we take the other element's field without merge logic

        Args:
            element1 (Tuple[Element, int]): the first element (and number of tokens in it)
            element2 (Tuple[Element, int]): the second element (and number of tokens in it)

        Returns:
            Tuple[Element, int]: a new merged element from the inputs (and number of tokens in it)
        """
        elt1, tok1 = element1
        elt2, tok2 = element2
        new_toks = 0
        new_elt = Element()
        new_elt.type = "Section"
        # Merge binary representations by concatenation
        if elt1.binary_representation is None or elt2.binary_representation is None:
            new_elt.binary_representation = elt1.binary_representation or elt2.binary_representation
        else:
            new_elt.binary_representation = elt1.binary_representation + elt2.binary_representation
        # Merge text representations by concatenation with a newline
        if elt1.text_representation is None or elt2.text_representation is None:
            new_elt.text_representation = elt1.text_representation or elt2.text_representation
            new_toks = max(tok1, tok2)
        else:
            new_elt.text_representation = elt1.text_representation + "\n" + elt2.text_representation
            new_toks = tok1 + 1 + tok2
        # Merge bbox by taking the coords that make the largest box
        if elt1.bbox is None and elt2.bbox is None:
            pass
        elif elt1.bbox is None or elt2.bbox is None:
            new_elt.bbox = elt1.bbox or elt2.bbox
        else:
            new_elt.bbox = BoundingBox(
                min(elt1.bbox.x1, elt2.bbox.x1),
                min(elt1.bbox.y1, elt2.bbox.y1),
                max(elt1.bbox.x2, elt2.bbox.x2),
                max(elt1.bbox.y2, elt2.bbox.y2),
            )
        # Merge properties by taking the union of the keys
        for k, v in elt1.properties.items():
            new_elt.properties[k] = v
        for k, v in elt2.properties.items():
            if new_elt.properties.get(k) is None:
                new_elt.properties[k] = v

        return (new_elt, new_toks)

    def merge_elements(self, document: Document) -> Document:
        """Use self._should_merge and self._merge to greedily merge consecutive elements.
        If the next element should be merged into the last 'accumulation' element, merge it.

        Args:
            document (Document): A document with elements to be merged.

        Returns:
            Document: The same document, with its elements merged
        """
        if len(document.elements) < 2:
            return document
        token_counts = [len(self.tokenizer.tokenize(e.text_representation or "")) for e in document.elements]
        new_elts = [(document.elements[0], token_counts[0])]
        for element, tokens in zip(document.elements[1:], token_counts[1:]):
            if self._should_merge(new_elts[-1], (element, tokens)):
                new_elts[-1] = self._merge(new_elts[-1], (element, tokens))
            else:
                new_elts.append((element, tokens))
        document.elements = [x[0] for x in new_elts]
        return document


class Merge(NonCPUUser, NonGPUUser, Transform):
    """
    Merge Elements into fewer large elements
    """

    def __init__(self, child: Node, merger: ElementMerger, **kwargs):
        super().__init__(child, **kwargs)
        self._merger = merger

    def execute(self) -> Dataset:
        input_dataset = self.child().execute()
        dataset = input_dataset.map(
            generate_map_class_from_callable(self._merger.merge_elements),
            compute=ActorPoolStrategy(min_size=1),
            **self.resource_args
        )
        return dataset
