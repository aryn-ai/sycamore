import re

from ray.data import Dataset

from sycamore.data import Document
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function
from sycamore.utils.time_trace import timetrace

COALESCE_WHITESPACE = [
    (r"\s+", " "),
    (r"^ ", ""),
    (r" $", ""),
]


class RegexReplace(SingleThreadUser, NonGPUUser, Transform):
    """
    The RegexReplace transform modifies the text_representation in each
    Element in every Document.

    Args:
        child: The source node or component that provides the documents
        spec: A list of tuples of regular expressions and substitutions,
              to be executed in order via re.sub()
        kwargs: Additional resource-related arguments that can be passed to the operation

    Example:
        .. code-block:: python

            rr = RegexReplace(child=node, spec=[(r"\s+", " "), (r"^ ", "")])
            dataset = rr.execute()
    """

    def __init__(self, child: Node, spec: list[tuple[str, str]], **kwargs):
        super().__init__(child, **kwargs)
        try:
            for x, y in spec:  # make sure it's iterable as pairs
                s = str()
                s += x  # only strings can be added to strings
                s += y
        except Exception:
            raise TypeError("RegexReplace spec is not list[tuple[str, str]]")
        self.spec = spec

    class Callable:
        def __init__(self, spec: list[tuple[str, str]]):
            self.spec = []
            for exp, repl in spec:
                pat = re.compile(exp)
                self.spec.append((pat, repl))

        @timetrace("regexRepl")
        def run(self, doc: Document) -> Document:
            spec = self.spec
            updated = []
            elements = doc.elements  # makes a copy
            for elem in elements:
                txt = elem.text_representation
                if txt is not None:
                    for rex, repl in spec:
                        txt = rex.sub(repl, txt)
                    elem.text_representation = txt
                    elem.binary_representation = txt.encode("utf-8")
                updated.append(elem)
            doc.elements = updated  # copies back
            return doc

    def execute(self) -> Dataset:
        ds = self.child().execute()
        xform = RegexReplace.Callable(self.spec)
        return ds.map(generate_map_function(xform.run))
