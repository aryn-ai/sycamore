import re


from sycamore.data import Document
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace

COALESCE_WHITESPACE = [
    (r"\s+", " "),
    (r"^ ", ""),
    (r" $", ""),
]


class RegexReplace(SingleThreadUser, NonGPUUser, Map):
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
        try:
            for x, y in spec:  # make sure it's iterable as pairs
                s = str()
                s += x  # only strings can be added to strings
                s += y
        except Exception:
            raise TypeError("RegexReplace spec is not list[tuple[str, str]]")

        compiled = []
        for exp, repl in spec:
            pat = re.compile(exp)
            compiled.append((pat, repl))

        @timetrace("regexRepl")
        def regex_replace(doc: Document) -> Document:
            for elem in doc.elements:
                txt = elem.text_representation
                if txt is not None:
                    for rex, repl in compiled:
                        txt = rex.sub(repl, txt)
                    elem.text_representation = txt
                    elem.binary_representation = txt.encode("utf-8")
            return doc

        super().__init__(child, f=regex_replace, **kwargs)
