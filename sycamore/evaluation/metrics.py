from abc import abstractmethod

from sycamore.evaluation import EvaluationDataPoint


class EvaluationMetric:
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def metric_name(self) -> str:
        pass

    @abstractmethod
    def evaluate(self, datapoint: EvaluationDataPoint) -> dict[str, str]:
        pass
