from sycamore.data import Document
from sycamore.functions.tokenizer import Tokenizer
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms import Map
from sycamore.utils.time_trace import timetrace

# TODO:
# - make breaks balanced in size
# - maybe move token counting elsewhere to avoid duplicate work


class MarkDropTiny(SingleThreadUser, NonGPUUser, Map):
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
        super().__init__(child, f=MarkDropTiny.mark_drop_tiny, args=[minimum], **resource_args)

    @staticmethod
    @timetrace("markDropTiny")
    def mark_drop_tiny(parent: Document, minimum) -> Document:
        for elem in parent.elements:
            tr = elem.text_representation or ""
            if len(tr) < minimum:
                elem.data["_drop"] = True  # remove specks
        return parent


###############################################################################


class MarkBreakPage(SingleThreadUser, NonGPUUser, Map):
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
        super().__init__(child, f=MarkBreakPage.mark_break_page, **resource_args)

    @staticmethod
    @timetrace("markBreakPage")
    def mark_break_page(parent: Document) -> Document:
        if len(parent.elements) > 1:
            last = parent.elements[0].properties["page_number"]
            for elem in parent.elements:
                page = elem.properties["page_number"]
                if page != last:
                    elem.data["_break"] = True  # mark for later
                    last = page
        return parent


###############################################################################


class MarkBreakByTokens(SingleThreadUser, NonGPUUser, Map):
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
        super().__init__(child, f=MarkBreakByTokens.mark_break_by_tokens, args=[tokenizer, limit], **resource_args)

    @staticmethod
    @timetrace("markBreakToks")
    def mark_break_by_tokens(parent: Document, tokenizer: Tokenizer, limit: int) -> Document:
        toks = 0
        for elem in parent.elements:
            if elem.text_representation:
                n = len(tokenizer.tokenize(elem.text_representation))
            else:
                n = 0
            elem.data["_tokCnt"] = n
            if elem.data.get("_break") or ((toks + n) > limit):
                elem.data["_break"] = True
                toks = 0
            toks += n
        return parent
