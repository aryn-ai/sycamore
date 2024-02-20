from abc import abstractmethod
from pathlib import Path
from typing import Any

from rouge import rouge
from sycamore.evaluation import EvaluationDataPoint
import logging

logger = logging.getLogger("ray")


class EvaluationMetric:
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def metric_name(self) -> str:
        pass

    @abstractmethod
    def evaluate(self, datapoint: EvaluationDataPoint) -> dict[str, Any]:
        pass


class DocumentRetrievalMetrics(EvaluationMetric):
    def __init__(self, recall_k: int = 10) -> None:
        super().__init__()
        self._recall_k = recall_k

    def metric_name(self) -> str:
        return "DocumentRetrievalMetrics"

    def evaluate(self, datapoint: EvaluationDataPoint) -> dict[str, Any]:
        result: dict[str, float] = {}
        correct_doc_count = 0
        correct_page_count = 0
        doc_mrr_sum = 0.0
        page_mrr_sum = 0.0
        logger.debug("Processing datapoint: " + str(datapoint))

        # we only support 1 source document currently
        for ground_truth_document in datapoint.ground_truth_source_documents[:1]:
            for i, document in enumerate(datapoint.generated_source_documents[: self._recall_k]):
                # only use filename currently, can enforce full url here if required
                doc_path = Path(document.properties["_location"]).name
                ground_truth_path = Path(ground_truth_document.properties["_location"]).name

                if doc_path == ground_truth_path:
                    result["correct_position"] = result.get("correct_position", i + 1)
                    doc_mrr_sum += 1.0 / (i + 1.0)
                    correct_doc_count += 1
                    if document.properties["page_number"] == ground_truth_document.properties["page_number"]:
                        result["correct_page_position"] = result.get("correct_page_position", i + 1)
                        page_mrr_sum += 1.0 / (i + 1.0)
                        correct_page_count += 1

        result["doc_recall"] = 1.0 if correct_doc_count > 0 else 0.0
        result["page_recall"] = 1.0 if correct_page_count > 0 else 0.0
        result["doc_mrr"] = min(1.0, doc_mrr_sum / len(datapoint.ground_truth_source_documents))
        result["page_mrr"] = min(1.0, page_mrr_sum / len(datapoint.ground_truth_source_documents))
        return result


class GeneratedAnswerMetrics(EvaluationMetric):
    def __init__(self, rouge_metrics=None) -> None:
        super().__init__()
        if rouge_metrics is None:
            rouge_metrics = ["rouge-1", "rouge-2", "rouge-l"]
        self._rouge_evaluator = rouge.Rouge(metrics=rouge_metrics)

    def metric_name(self) -> str:
        return "GeneratedAnswerMetrics"

    def evaluate(self, datapoint: EvaluationDataPoint) -> dict[str, str]:
        scores = self._rouge_evaluator.get_scores(datapoint.generated_answer, datapoint.ground_truth_answer)[0]
        result = {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
        return result


document_retrieval_metrics = DocumentRetrievalMetrics()
generated_answer_metrics = GeneratedAnswerMetrics()
