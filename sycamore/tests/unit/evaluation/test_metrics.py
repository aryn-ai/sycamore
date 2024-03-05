from rouge import rouge

from sycamore.data import Element
from sycamore.evaluation import EvaluationDataPoint
from sycamore.evaluation.metrics.retrieval import DocumentRetrievalMetrics
from sycamore.evaluation.metrics.generated_answer import RougeMetrics


def test_document_retrieval_metrics():
    recall_k = 2
    metrics = DocumentRetrievalMetrics(recall_k=recall_k)

    # Enforce metric name is class name
    assert metrics.metric_name() == "DocumentRetrievalMetrics"

    datapoint = EvaluationDataPoint(
        {
            "ground_truth_source_documents": [Element({"properties": {"_location": "file1.pdf", "page_number": 10}})],
            "generated_source_documents": [
                Element({"properties": {"_location": "file1.pdf", "page_number": 3}}),
                Element({"properties": {"_location": "http://absolutepath/dir/file1.pdf", "page_number": 10}}),
                Element({"properties": {"_location": "file2.pdf", "page_number": 1}}),
            ],
        }
    )
    result = metrics.evaluate(datapoint)
    assert result["doc_recall"] == 1
    assert result["page_recall"] == 1
    assert result["doc_mrr"] == 1
    assert result["page_mrr"] == 0.5

    result = DocumentRetrievalMetrics(recall_k=1).evaluate(datapoint)

    assert result["doc_recall"] == 1
    assert result["page_recall"] == 0
    assert result["doc_mrr"] == 1
    assert result["page_mrr"] == 0


def test_document_retrieval_metrics_multi_page_indexed():
    recall_k = 2
    metrics = DocumentRetrievalMetrics(recall_k=recall_k)

    # Enforce metric name is class name
    assert metrics.metric_name() == "DocumentRetrievalMetrics"

    datapoint = EvaluationDataPoint(
        {
            "ground_truth_source_documents": [Element({"properties": {"_location": "file1.pdf", "page_number": 3}})],
            "generated_source_documents": [
                Element({"properties": {"_location": "file1.pdf", "page_numbers": [1, 2]}}),
                Element({"properties": {"_location": "file1.pdf", "page_numbers": [3, 4]}}),
            ],
        }
    )
    result = metrics.evaluate(datapoint)
    assert result["doc_recall"] == 1
    assert result["page_recall"] == 1
    assert result["doc_mrr"] == 1
    assert result["page_mrr"] == 0.5

    result = DocumentRetrievalMetrics(recall_k=1).evaluate(datapoint)

    assert result["doc_recall"] == 1
    assert result["page_recall"] == 0
    assert result["doc_mrr"] == 1
    assert result["page_mrr"] == 0


def test_document_retrieval_metrics_multi_page_indexed_and_gold():
    metrics = DocumentRetrievalMetrics()

    # Enforce metric name is class name
    assert metrics.metric_name() == "DocumentRetrievalMetrics"

    datapoint = EvaluationDataPoint(
        {
            "ground_truth_source_documents": [
                Element({"properties": {"_location": "file1.pdf", "page_number": 3, "page_numbers": [3, 4]}})
            ],
            "generated_source_documents": [
                Element({"properties": {"_location": "file1.pdf", "page_numbers": [1, 2]}}),
                Element({"properties": {"_location": "file1.pdf", "page_numbers": [2, 3]}}),
                Element({"properties": {"_location": "file1.pdf", "page_numbers": [3, 4]}}),
            ],
        }
    )
    result = metrics.evaluate(datapoint)
    assert result["doc_recall"] == 1
    assert result["page_recall"] == 1
    assert result["partial_page_recall"] == 1
    assert result["doc_mrr"] == 1
    assert result["partial_page_mrr"] == 0.5 + (1 / 3)  # 2nd and 3rd partially match
    assert result["page_mrr"] == 1 / 3

    result = DocumentRetrievalMetrics(recall_k=2).evaluate(datapoint)

    assert result["doc_recall"] == 1
    assert result["page_recall"] == 0
    assert result["partial_page_recall"] == 1
    assert result["doc_mrr"] == 1
    assert result["page_mrr"] == 0
    assert result["partial_page_mrr"] == 0.5  # 2nd partially matches


def test_generated_answer_metrics():
    rouge_impl = rouge.Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    metrics = RougeMetrics()
    # Enforce metric name is class name
    assert metrics.metric_name() == "GeneratedAnswerMetrics"

    datapoint = EvaluationDataPoint(
        {
            "generated_answer": "The color of the sky is blue unless it's cloudy.",
            "ground_truth_answer": "The sky is blue when there are no clouds",
        }
    )
    scores = rouge_impl.get_scores(datapoint.generated_answer, datapoint.ground_truth_answer)[0]

    result = metrics.evaluate(datapoint)

    assert result["rouge-1"] == scores["rouge-1"]["f"]
    assert result["rouge-2"] == scores["rouge-2"]["f"]
    assert result["rouge-l"] == scores["rouge-l"]["f"]
