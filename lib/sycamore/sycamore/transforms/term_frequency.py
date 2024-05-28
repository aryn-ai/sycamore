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

    Args:
        child: Source node that provides documents to compute TF for.
        tokenizer: The Tokenizer object to use to split words in order to count them
        with_token_ids: Create the TF table using token_ids (True) or token values (False)
                default is False (toekn values)

    Example:
        .. code-block:: python

            tk = OpenAITokenizer("gpt-3.5-turbo")
            context = sycamore.init()
            context.read.binary(paths, binary_format="pdf")
                .partition(SycamorePartitioner())
                .explode()
                .transform(cls=TermFrequency, tokenizer=tk)
                .show()

    """

    def __init__(self, child: Node, tokenizer: Tokenizer, with_token_ids: bool = False, **kwargs):
        super().__init__(child, f=compute_term_frequency, args=[tokenizer, with_token_ids], **kwargs)
