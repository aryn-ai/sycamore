from typing import Dict, List, Union, Any, Tuple
from abc import ABC, abstractmethod

import os
from sycamore.data import Element, Document
from sycamore.evaluation.pipeline import EvaluationPipeline
from sycamore.evaluation import EvaluationDataPoint, EvaluationMetric
from sycamore.evaluation.metrics import document_retrieval_metrics
from sycamore import Context


def add_search_context_to_datapoint(datapoint: Document) -> List[Element]:
    """
    This function adds raw information from JSON ground truth and properties
    to each evaluation datapoint.
    """
    assert datapoint.type == "EvaluationDataPoint"
    source_documents: List[Element] = []
    assert datapoint.get("raw") is not None
    for search_result in datapoint.get("raw", {}).get("SearchContexts", []):
        source_document = Element()
        pg_number = search_result.get("page_numbers", [search_result.get("page_number", None)])
        assert pg_number is not None
        properties = {
            "_location": search_result.get("document_url", ""),
            "page_number": pg_number[0],
            "doc_id": search_result.get("document_id", [0]),
        }
        source_document.properties = properties
        source_document.text_representation = search_result["text_representation"]
        source_documents += [source_document]
    return source_documents


def add_filters_to_question(datapoint: EvaluationDataPoint) -> EvaluationDataPoint:
    """
    This function adds properties, search context and custom question augmentation to
    evaluation datapoints.
    """
    assert datapoint.type == "EvaluationDataPoint"
    datapoint = EvaluationDataPoint(datapoint)
    assert datapoint.raw is not None
    template_Q = datapoint.raw["custom_question_augmentation"]
    filter_Q = datapoint.raw["question_augmentation_filter"]
    if datapoint.filters:
        for filter in datapoint.filters.keys():
            if datapoint.raw.get("Filters") is not None:
                datapoint.filters[filter] = datapoint.raw.get("Filters").get(filter, "")
        datapoint.ground_truth_source_documents = add_search_context_to_datapoint(datapoint)
        datapoint.question = template_Q.format(datapoint.raw["Question"], datapoint.filters.get(filter_Q))
    return datapoint


class Assessment(ABC):
    @abstractmethod
    def run_evaluation(self, ctx: Context, index: str, **kwargs):
        pass

    def __call__(self, context: Context, index: str, **kwargs):
        return self.run_evaluation(context, index, **kwargs)


class QualityAssessment(Assessment):
    """
    This Class runs Question Answering benchmarking on each datapoint and
    measure performance.

    Attributes:
        gt_path: Path to ground truth, supports JSON format.
        rag_config: Configration for RAG.
        os_client_args: Configration for connecting to opensearch.
        custom_question_augmentation: Custom String for Augmenting Question.
        question_augmentation_filter: Filters values to be use in custom Question Augmentation.
        metrics: Metrics to test score on, eg document_retrieval_metrics.
        custom_question_augmentation: A string to format questions.
        question_augmentation_filter: A Filter string to be used in formating question.
    """

    def __init__(
        self,
        gt_path: str,
        rag_config: Dict[str, Any],
        os_client_args: Dict[str, Any],
        metrics: List[EvaluationMetric] = [document_retrieval_metrics],
        custom_question_augmentation: str = "{}",
        question_augmentation_filter: str = "",
    ):
        self.user = os.environ.get("USER", os.environ.get("USERNAME"))
        self.gt_path = gt_path
        self.rag_config = rag_config
        self.os_client_args = os_client_args
        self.metrics = metrics
        self.custom_question_augmentation = custom_question_augmentation
        self.question_augmentation_filter = question_augmentation_filter

    @staticmethod
    def create_evaluation_datapoint(
        json_dict: Dict, custom_question_augmentation: str = "{}", question_augmentation_filter: str = ""
    ) -> List[Dict[str, bytes]]:
        """
        This method creates evaluation datapoint from JSON file.
        """
        result = []
        assert json_dict is not None
        assert isinstance(json_dict, dict)
        for datapoint in json_dict["data"][:]:
            document = EvaluationDataPoint()
            datapoint["custom_question_augmentation"] = custom_question_augmentation
            datapoint["question_augmentation_filter"] = question_augmentation_filter
            document.raw = datapoint
            document.ground_truth_answer = datapoint["Answer"]
            document.filters = datapoint.get("Filters", {})
            result += [{"doc": document.serialize()}]
        return result

    def run_evaluation(self, ctx: Context, index: str, **kwargs) -> Tuple[List[Document], Document]:
        """
        This method runs evaluation test by following steps:
        1. Creating evaluation datapoint.
        2. Creating evaluation pipeline.
        3. Using datapoint on pipeline for evaluation.
        """
        custom_question_augmentation = str(self.custom_question_augmentation)
        question_augmentation_filter = str(self.question_augmentation_filter)
        input_docset = ctx.read.json(
            paths=self.gt_path,
            doc_extractor=lambda json_dict: QualityAssessment.create_evaluation_datapoint(
                json_dict, custom_question_augmentation, question_augmentation_filter
            ),
        ).map(add_filters_to_question)

        pipeline = EvaluationPipeline(
            index=index, os_client_args=self.os_client_args, os_config=self.rag_config, metrics=self.metrics
        )
        query_level_metrics, aggregated_metrics = pipeline.execute(input_docset)
        return query_level_metrics.take_all(), aggregated_metrics


class Evaluate:
    """
    The Evaluate runs the evaluation test on
    index or list of indices against a ground truth.

    Args:
        context: The Sycamore Context to use.
        index: Index or list of Index.
        assessment: The Assessment to run, there is only one right now.

    Returns:
        Two EvaluationDataPoint, one for query level information and another with aggregate information.

    Example:
        context = sycamore.init()

        custom_question_augmentation = "{}, The product code is {}."
        question_augmentation_filter = 'properties._product_codes'

        assessment = QualityAssessment(os_client_args=OS_CLIENT_ARGS,
            rag_config= OS_CONFIG,
            gt_path = './test.json',
            custom_question_augmentation=custom_question_augmentation,
            question_augmentation_filter = question_augmentation_filter)
        evaluate = Evaluate(context,'index_V1',assessment).run()
    """

    def __init__(self, context: Context, index: Union[str, List[str]], assessment: Assessment):
        self.context = context
        self.index = index
        self.assessment = assessment
        self._result = None
        super().__init__()

    @property
    def result(self):
        return self._result

    def run(self):
        if isinstance(self.index, str):
            self._result = {self.index: self.assessment(self.context, self.index)}
        elif isinstance(self.index, List) and all(isinstance(i, str) for i in self.index):
            self._result = {idx: self.assessment(self.context, idx) for idx in self.index}
        else:
            raise ValueError("Input must be a str or a list of str")
