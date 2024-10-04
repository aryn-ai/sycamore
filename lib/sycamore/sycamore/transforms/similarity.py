import logging
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from sycamore.data.document import DocumentPropertyTypes, DocumentSource
from sycamore.utils.import_utils import requires_modules

from sycamore.data import Document, Element
from sycamore.plan_nodes import Node
from sycamore.transforms import MapBatch
from sycamore.utils import choose_device
from sycamore.utils.time_trace import timetrace

if TYPE_CHECKING:
    pass
logger = logging.getLogger(__name__)


class SimilarityScorer(ABC):
    """
    SimilarityScorer is an abstract class to compute similarity scores between a query string and a Document's
    contents. It requires a child class to implement the score(..) method to compute similarity scores for a batch
    of pairs of strings.

    Currently, it will score each element in each provided document (unless ignores via ignore_element_sources).
    It will also propagate the highest score to a document level property along with the element_id of the source.

    Args:
        ignore_element_sources: Ignore elements if they belong to these sources
        ignore_doc_structure: Ignore Document model (Document->Elements) in a DocSet
    """

    def __init__(
        self, ignore_element_sources: Optional[list[DocumentSource]] = None, ignore_doc_structure: bool = False
    ):
        if ignore_element_sources is None:
            ignore_element_sources = [DocumentSource.DOCUMENT_RECONSTRUCTION_RETRIEVAL]
        self._ignore_element_sources = ignore_element_sources
        self._ignore_doc_structure = ignore_doc_structure

    def __call__(self, doc_batch: list[Document], query: str, score_property_name: str) -> list[Document]:
        return self.generate_similarity_scores(doc_batch, query, score_property_name)

    def _get_inputs_from_document(self, document: Document) -> list[Element]:
        if self._ignore_doc_structure:
            return [Element(document.data)] if document.text_representation else []

        result: list[Element] = list()
        for element in document.elements:
            assert element.element_index is not None, "generating similarity score requires element_index"
            if (
                element.properties.get(DocumentPropertyTypes.SOURCE, "") not in self._ignore_element_sources
                and element.text_representation
            ):
                result.append(element)
        return result

    def _populate_score(self, score: float, score_property_name: str, document: Document, element: Element) -> Document:
        if self._ignore_doc_structure:
            document.properties[score_property_name] = score
            return document
        assert element.element_index is not None, "populating similarity score requires the element's index"
        element.properties[score_property_name] = score

        doc_score = document.properties.get(score_property_name, float("-inf"))
        if score > doc_score:
            document.properties[score_property_name] = score
            document.properties[f"{score_property_name}_source_element_index"] = element.element_index
        return document

    def generate_similarity_scores(
        self, doc_batch: list[Document], query: str, score_property_name: str
    ) -> list[Document]:

        input_metadata: list[tuple[int, Element]] = []
        input_pairs = []

        for doc_idx, doc in enumerate(doc_batch):
            candidate_elements = self._get_inputs_from_document(doc)
            for element in candidate_elements:
                assert element.text_representation is not None, "Found element without text_representation"
                input_pairs.append((query, element.text_representation))
                input_metadata.append((doc_idx, element))

        if not input_pairs:
            return doc_batch

        scores = self.score(input_pairs)

        for i, (doc_idx, element) in enumerate(input_metadata):
            self._populate_score(scores[i], score_property_name, doc_batch[doc_idx], element)

        return doc_batch

    @abstractmethod
    def score(self, inputs: list[tuple[str, str]]) -> list[float]:
        pass


class HuggingFaceTransformersSimilarityScorer(SimilarityScorer):
    """
    HuggingFaceTransformersSimilarityScorer is an SimilarityScorer class for generating sentence similarity using the
    transformers package.

    Args:
        model_name: Name or path of the Transformers model to use for similarity scoring.
        model_batch_size: Batch size used by the underlying Transformers model for similarity scoring, default is 16.
        max_tokens: Max tokens to use for tokenization, default is 512.
        device: Device (e.g., "cpu" or "cuda") on which to perform embedding.
        ignore_doc_structure: Ignore Document model (Document->Elements) in a DocSet

    Example:
        .. code-block:: python

            similarity_scorer = HuggingFaceTransformersSimilarityScorer(ignore_doc_structure=True)

            doc = Document(
                {
                    "doc_id": 1,
                    "elements": [
                        {"text_representation": "here is an animal with 4 legs and whiskers"},
                    ],
                }
            )
            result = similarity_scorer.generate_similarity_scores(
                [doc], query="is this a cat?", score_property_name="similarity_score"
            )
            print(result.properties["similarity_score"])

    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        model_batch_size: int = 16,
        max_tokens: int = 512,
        device: Optional[str] = None,
        ignore_doc_structure: bool = False,
        ignore_element_sources: Optional[list[DocumentSource]] = None,
    ):
        super().__init__(ignore_element_sources=ignore_element_sources, ignore_doc_structure=ignore_doc_structure)
        self.device = choose_device(device)
        self.model_name = model_name
        self.model_batch_size = model_batch_size
        self.max_tokens = max_tokens

        self._model = None
        self._tokenizer = None

    @requires_modules(["transformers", "torch"], extra="local-inference")
    def __call__(self, doc_batch: list[Document], query: str, score_property_name: str) -> list[Document]:
        return self.generate_similarity_scores(doc_batch, query, score_property_name)

    @timetrace("TransformersSimilarity")
    def score(self, inputs: list[tuple[str, str]]) -> list[float]:
        import torch

        if not self._model or not self._tokenizer:
            logger.info(f"Initializing transformers model: {self.model_name}")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)

        assert self._model is not None
        assert self._tokenizer is not None

        scores = []
        with torch.no_grad():
            for i in range(0, len(inputs), self.model_batch_size):
                input_batch = inputs[i : i + self.model_batch_size]

                tokenized = self._tokenizer(
                    input_batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_tokens
                )
                scores.extend(
                    (
                        self._model(**tokenized, return_dict=True)
                        .logits.view(
                            -1,
                        )
                        .float()
                    )
                )
            return scores


class ScoreSimilarity(MapBatch):
    """
    ScoreSimilarity is a Map class for executing a similarity scoring function on Documents. These
    similarity scores can then be used to rank documents for search use cases, or filter irrelevant documents.

    Args:
        child: The source node or component that provides the dataset containing text data.
        similarity_scorer: An instance of an SimilarityScorer class that executes the scoring function.
        query: The query string to compute similarity against.
        score_property_name: The name of the key where the score will be stored in document.properties
        resource_args: Additional resource-related arguments that can be passed to the extraction operation.

    Example:
         .. code-block:: python

            source_node = ...  # Define a source node or component that provides a dataset with text data.
            custom_scorer = MyScoringTechnique(entity_extraction_params)
            scoring_transform = ScoreSimilarity(child=source_node, similarity_scorer=custom_scorer, query="Cats?")
            scored_dataset = scoring_transform.execute()

    """

    def __init__(
        self,
        child: Node,
        similarity_scorer: SimilarityScorer,
        query: str,
        score_property_name: str = "_similarity_score",
        **resource_args,
    ):
        super().__init__(
            child,
            f=similarity_scorer,  # type: ignore
            f_args=[query, score_property_name],
            **resource_args,
        )
