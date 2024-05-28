from collections import Counter
from sycamore.data.document import Document
from sycamore.functions.tokenizer import Tokenizer
from sycamore.plan_nodes import Node, NonGPUUser, SingleThreadUser, Transform
from sycamore.transforms.map import Map


def compute_term_frequency(document: Document, tokenizer: Tokenizer, with_token_ids: bool) -> Document:
    tokens = tokenizer.tokenize(document.text_representation or "", as_ints=with_token_ids)
    table = dict(Counter(tokens))
    document.properties["term_frequency"] = table
    return document


class TermFrequency(SingleThreadUser, NonGPUUser, Map, Transform):
    """
    Generate a table of frequencies of terms in the text representation of each document
    """

    def __init__(self, child: Node, tokenizer: Tokenizer, with_token_ids: bool = False, **kwargs):
        super().__init__(child, f=compute_term_frequency, args=[tokenizer, with_token_ids], **kwargs)
