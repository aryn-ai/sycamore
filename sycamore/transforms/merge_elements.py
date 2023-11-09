from abc import ABC, abstractmethod

from ray.data import Dataset, ActorPoolStrategy

from sycamore.data import Document, Element, BoundingBox
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Transform, Node
from sycamore.utils import generate_map_class_from_callable
from sycamore.functions.tokenizer import Tokenizer


class ElementMerger(ABC):
    @abstractmethod
    def should_merge(self, element1: Element, element2: Element) -> bool:
        pass

    @abstractmethod
    def merge(self, element1: Element, element2: Element) -> Element:
        pass

    @abstractmethod
    def preprocess_element(self, element: Element) -> Element:
        pass

    @abstractmethod
    def postprocess_element(self, element: Element) -> Element:
        pass

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
        to_merge = [self.preprocess_element(e) for e in document.elements]
        new_elements = [to_merge[0]]
        for element in to_merge[1:]:
            if self.should_merge(new_elements[-1], element):
                new_elements[-1] = self.merge(new_elements[-1], element)
            else:
                new_elements.append(element)
        document.elements = [self.postprocess_element(e) for e in new_elements]
        return document


class GreedyTextElementMerger(ElementMerger):
    def __init__(self, tokenizer: Tokenizer, max_tokens: int):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def preprocess_element(self, element: Element) -> Element:
        element.data["token_count"] = len(self.tokenizer.tokenize(element.text_representation or ""))
        return element

    def postprocess_element(self, element: Element) -> Element:
        del element.data["token_count"]
        return element

    def should_merge(self, element1: Element, element2: Element) -> bool:
        if element1.data["token_count"] + 1 + element2.data["token_count"] > self.max_tokens:
            return False
        return True

    def merge(self, elt1: Element, elt2: Element) -> Element:
        """Merge two elements; the new element's fields will be set as:
            - type: "Section"
            - binary_representation: elt1.binary_representation + elt2.binary_representation
            - text_representation: elt1.text_representation + elt2.text_representation
            - bbox: the minimal bbox that contains both elt1's and elt2's bboxes
            - properties: elt1's properties + any of elt2's properties that are not in elt1
            note: if elt1 and elt2 have different values for the same property, we take elt1's value
            note: if any input field is None we take the other element's field without merge logic

        Args:
            element1 (Tuple[Element, int]): the first element (and number of tokens in it)
            element2 (Tuple[Element, int]): the second element (and number of tokens in it)

        Returns:
            Tuple[Element, int]: a new merged element from the inputs (and number of tokens in it)
        """
        tok1 = elt1.data["token_count"]
        tok2 = elt2.data["token_count"]
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
            new_elt.data["token_count"] = max(tok1, tok2)
        else:
            new_elt.text_representation = elt1.text_representation + "\n" + elt2.text_representation
            new_elt.data["token_count"] = tok1 + 1 + tok2
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
        properties = new_elt.properties
        for k, v in elt1.properties.items():
            properties[k] = v
        for k, v in elt2.properties.items():
            if properties.get(k) is None:
                properties[k] = v
        new_elt.properties = properties

        return new_elt


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
