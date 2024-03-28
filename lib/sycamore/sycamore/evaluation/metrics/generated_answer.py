from rouge import rouge

from sycamore.evaluation import EvaluationDataPoint, EvaluationMetric


class RougeMetrics(EvaluationMetric):
    def __init__(self, rouge_metrics_types=None) -> None:
        super().__init__()
        if rouge_metrics_types is None:
            rouge_metrics_types = ["rouge-1", "rouge-2", "rouge-l"]
        self._rouge_evaluator = rouge.Rouge(metrics=rouge_metrics_types)

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


rouge_metrics = RougeMetrics()
