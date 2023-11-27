from ray.data import Dataset

from sycamore.data import Document, Element
from sycamore.functions.tokenizer import Tokenizer
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function


class SplitElements(SingleThreadUser, NonGPUUser, Transform):
    """
    The SplitElements transform recursively divides elements such that no
    Element exceeds a maximum number of tokens.

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
                result.extend(self.splitUp(elem))
            parent.elements = result
            return parent

        def splitUp(self, elem: Element) -> list[Element]:
            txt = elem.text_representation
            if not txt:
                return [elem]
            num = len(self.tokenizer.tokenize(txt))
            if num <= self.max:
                return [elem]

            half = len(txt) // 2
            left = half
            right = half + 1

            # FIXME: make this work with asian languages
            predicates = [  # in precedence order
                lambda c: c in ".!?",
                lambda c: c == ";",
                lambda c: c in "()",
                lambda c: c == ":",
                lambda c: c == ",",
                str.isspace,
            ]
            results: list[int | None] = [None] * len(predicates)

            for jj in range(half // 2):  # stay near middle; avoid the ends
                lchar = txt[left]
                rchar = txt[right]

                go = True
                for ii, predicate in enumerate(predicates):
                    if predicate(lchar):
                        if results[ii] is None:
                            results[ii] = left
                        go = ii != 0
                        break
                    elif predicate(rchar):
                        if results[ii] is None:
                            results[ii] = right
                        go = ii != 0
                        break
                if not go:
                    break

                left -= 1
                right += 1

            idx = half + 1
            for res in results:
                if res is not None:
                    idx = res + 1
                    break

            one = txt[:idx]
            two = txt[idx:]

            ment = elem.copy()
            elem.text_representation = one
            elem.binary_representation = bytes(one, "utf-8")
            ment.text_representation = two
            ment.binary_representation = bytes(two, "utf-8")
            aa = self.splitUp(elem)
            bb = self.splitUp(ment)
            aa.extend(bb)
            return aa

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        xform = SplitElements.Callable(self.tokenizer, self.max)
        return dataset.map(generate_map_function(xform.run))
