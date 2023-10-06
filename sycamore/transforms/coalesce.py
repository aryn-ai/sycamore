from abc import ABC, abstractmethod

from sycamore.plan_nodes import Transform, Node
from sycamore.functions import CharacterTokenizer, Tokenizer
from sycamore.data import Document
from sycamore.transforms.map import generate_map_function


class Coalescer(ABC):
    @abstractmethod
    def coalesce(self, document: Document) -> Document:
        pass


class BBoxCoalescer(Coalescer):
    """
    BBoxCoalescer uses a greedy approach to merge Elements within a Document. It will only attempt to do this for
    documents where each element contains a 'bbox'.

    Args:
        tokenizer: Tokenizer implementation used to count and combine tokens into new elements
        max_tokens_per_element: The maximum text tokens an element can contain.

    Example:
         .. code-block:: python

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner(), table_extractor=table_extractor)
                .coalesce(coalescer=BBoxCoalescer(tokenizer=CharacterTokenizer(), max_tokens_per_element=1000))
    """
    def __init__(self, tokenizer: Tokenizer = CharacterTokenizer(), max_tokens_per_element=100) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._max_tokens_per_element = max_tokens_per_element

    def coalesce(self, document: Document) -> Document:
        # todo: implement
        return document

class Coalesce(Transform):
    """
    Coalesce is a transformation class for combining elements within a Document using a Coalescer.

    This transform condenses small elements within a document, e.g. list items, into a single element.

    Args:
        child: The source node or component that provides the dataset containing text data.
        coalscer: An instance of an Coalescer class that defines the coalescing to beapplied.
        resource_args: Additional resource-related arguments that can be passed to the extraction operation.

    Example:
         .. code-block:: python

            source_node = ...  # Define a source node or component that provides a dataset with text data.
            custom_coalscer = MyCoalescer(params)
            coalesce_transform = Coalesce(child=source_node, coalscer=custom_coalscer)
            coalesced_dataset = coalesce_transform.execute()

    """

    def __init__(
            self,
            child: Node,
            coalscer: Coalescer,
            **resource_args,
    ):
        super().__init__(child, **resource_args)
        self._coalscer = coalscer

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        dataset = input_dataset.map(generate_map_function(self._coalscer.coalesce()))
        return dataset
