from ray.data import Dataset

from sycamore.data import Document
from sycamore.functions.tokenizer import Tokenizer
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function
from sycamore.utils.time_trace import timetrace

# TODO:
# - make breaks balanced in size
# - maybe move token counting elsewhere to avoid duplicate work


class MarkDropTiny(SingleThreadUser, NonGPUUser, Transform):
    """
    MarkDropTiny is a transform to add the '_drop' data attribute to
    each Element smaller than a certain size.

    Args:
        child: The source Node or component that provides the Elements
        minimum: The smallest Element to keep (def 2)

    Example:
        .. code-block:: python

            source_node = ...
            marker = MarkDropTiny(child=source_node, minimum=2)
            dataset = marker.execute()
    """

    def __init__(self, child: Node, minimum: int = 2, **resource_args):
        super().__init__(child, **resource_args)
        self.min = minimum

    class Callable:
        def __init__(self, minimum: int):
            self.min = minimum

        @timetrace("markDropTiny")
        def run(self, parent: Document) -> Document:
            elements = parent.elements  # makes a copy
            for elem in elements:
                tr = elem.text_representation or ""
                if len(tr) < self.min:
                    elem.data["_drop"] = True  # remove specks
            parent.elements = elements  # copy back
            return parent

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        marker = MarkDropTiny.Callable(self.min)
        return dataset.map(generate_map_function(marker.run))


###############################################################################


class MarkBreakPage(SingleThreadUser, NonGPUUser, Transform):
    """
    MarkBreakPage is a transform to add the '_break' data attribute to
    each Element when the 'page_number' property changes.

    Args:
        child: The source Node or component that provides the Elements

    Example:
        .. code-block:: python

            source_node = ...
            marker = MarkBreakPage(child=source_node)
            dataset = marker.execute()
    """

    def __init__(self, child: Node, **resource_args):
        super().__init__(child, **resource_args)

    class Callable:
        @timetrace("markBreakPage")
        def run(self, parent: Document) -> Document:
            if len(parent.elements) > 1:
                elements = parent.elements  # makes a copy
                last = elements[0].properties["page_number"]
                for elem in elements:
                    page = elem.properties["page_number"]
                    if page != last:
                        elem.data["_break"] = True  # mark for later
                        last = page
                parent.elements = elements  # copy back
            return parent

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        marker = MarkBreakPage.Callable()
        return dataset.map(generate_map_function(marker.run))


###############################################################################


class MarkBreakByTokens(SingleThreadUser, NonGPUUser, Transform):
    """
    MarkBreakByTokens is a transform to add the '_break' data attribute to
    each Element when the number of tokens exceeds the limit.  This should
    most likely be the last marking operation before final merge.

    Args:
        child: The source Node or component that provides the Elements
        tokenizer: the tokenizer that will be used for embedding
        limit: maximum permitted number of tokens

    Example:
        .. code-block:: python

            source_node = ...
            marker = MarkBreakByTokens(child=source_node, limit=512)
            dataset = marker.execute()
    """

    def __init__(self, child: Node, tokenizer: Tokenizer, limit: int = 512, **resource_args):
        super().__init__(child, **resource_args)
        self.tokenizer = tokenizer
        self.limit = limit

    class Callable:
        def __init__(self, tokenizer: Tokenizer, limit: int):
            self.tokenizer = tokenizer
            self.limit = limit

        @timetrace("markBreakToks")
        def run(self, parent: Document) -> Document:
            toks = 0
            elements = parent.elements  # makes a copy
            for elem in elements:
                if elem.text_representation:
                    n = len(self.tokenizer.tokenize(elem.text_representation))
                else:
                    n = 0
                elem.data["_tokCnt"] = n
                if elem.data.get("_break") or ((toks + n) > self.limit):
                    elem.data["_break"] = True
                    toks = 0
                toks += n
            parent.elements = elements  # copy back
            return parent

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        marker = MarkBreakByTokens.Callable(self.tokenizer, self.limit)
        return dataset.map(generate_map_function(marker.run))
