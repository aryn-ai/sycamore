from ray.data import Dataset

from sycamore.data import Document, Element
from sycamore.functions.tokenizer import Tokenizer
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function


class SplitElements(SingleThreadUser, NonGPUUser, Transform):
    """
    The SplitElements transform divides elements such that no Element
    exceeds a maximum number of tokens.

    Args:
        child: The source node or component that provides the elements to be split
        tokenizer: The tokenizer to use in counting tokens, should match embedder
        maximum: Most tokens allowed in any Element

    Example:
        .. code-block:: python

            node = ...  # Define a source node or component that provides hierarchical documents.
            xform = SplitElements(child=node, tokenizer=tokenizer, 512)
            dataset = xform.execute()
    """

    def __init__(self, child: Node, tokenizer: Tokenizer, maximum: int, **kwargs):
        super().__init__(child, **kwargs)
        self.tokenizer = tokenizer
        self.max = maximum

    class Callable:
        def __init__(self, tokenizer, maximum):
            self.tokenizer = tokenizer
            self.max = maximum

        def run(self, parent: Document) -> Document:
            result = []
            elements = parent.elements  # makes a copy
            for elem in elements:
                if elem.text_representation:
                    splits = self.splitUp(elem)
                    result.extend(splits)
                else:
                    result.append(elem)
            parent.elements = result
            return parent

        def splitUp(self, elem: Element) -> list[Element]:
            txt = elem.text_representation
            num = len(self.tokenizer.tokenize(txt))
            if num <= self.max:
                return [elem]

            half = len(txt) // 2
            left = half
            right = half + 1
            period = None
            semi = None
            comma = None
            space = None

            for i in range(half - 2):  # avoid the ends
                ll = txt[left]
                rr = txt[right]
                if ll == '.':
                    period = left
                    break
                elif rr == '.':
                    period = right
                    break
                elif ll == ';':
                    if semi is None:
                        semi = left
                elif rr == ';':
                    if semi is None:
                        semi = right
                elif ll == ',':
                    if comma is None:
                        comma = left
                elif rr == ',':
                    if comma is None:
                        comma = right
                elif ll.isspace():
                    if space is None:
                        space = left
                elif rr.isspace():
                    if space is None:
                        space = right
                left -= 1
                right += 1

            idx = half
            if period is not None:
                idx = period + 1
            elif semi is not None:
                idx = semi + 1
            elif comma is not None:
                idx = comma + 1
            elif space is not None:
                idx = space + 1

            one = txt[ : idx]
            two = txt[idx : ]

            elem2 = elem.copy()
            elem.text_representation = one
            elem.binary_representation = bytes(one, "utf-8")
            elem2.text_representation = two
            elem2.binary_representation = bytes(two, "utf-8")
            aa = self.splitUp(elem)
            bb = self.splitUp(elem2)
            aa.extend(bb)
            return aa

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        xform = SplitElements.Callable(self.tokenizer, self.max)
        return dataset.map(generate_map_function(xform.run))
