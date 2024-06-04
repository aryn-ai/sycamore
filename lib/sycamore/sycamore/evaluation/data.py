from abc import abstractmethod
from typing import Optional, Any

from sycamore.data import Element
from sycamore.data import Document


class EvaluationDataPoint(Document):
    def __init__(
        self,
        document=None,
        **kwargs,
    ):
        super().__init__(document, **kwargs)
        self.data["type"] = "EvaluationDataPoint"

    @property
    def question(self) -> Optional[str]:
        """Natural language question."""
        return self.data.get("question")

    @question.setter
    def question(self, value: str) -> None:
        """Set the Natural language question."""
        self.data["question"] = value

    @property
    def ground_truth_answer(self) -> Optional[str]:
        """Natural language answer."""
        return self.data.get("ground_truth_answer")

    @ground_truth_answer.setter
    def ground_truth_answer(self, value: str) -> None:
        """Set the Natural language answer."""
        self.data["ground_truth_answer"] = value

    @property
    def filters(self) -> Optional[dict[str, str]]:
        """Filters to use for querying"""
        return self.data.get("filters")

    @filters.setter
    def filters(self, value: dict[str, str]) -> None:
        """Set the filters to use for querying"""
        self.data["filters"] = value

    @property
    def ground_truth_source_documents(self) -> list[Element]:
        """List of documents required to answer this question."""
        return self.data.get("ground_truth_source_documents", [])

    @ground_truth_source_documents.setter
    def ground_truth_source_documents(self, value: list[Element]) -> None:
        """Set the list of documents required to answer this question"""
        self.data["ground_truth_source_documents"] = value

    @property
    def generated_answer(self) -> Optional[str]:
        """Natural language generated answer."""
        return self.data.get("generated_answer")

    @generated_answer.setter
    def generated_answer(self, value: str) -> None:
        """Set the Natural language generated answer."""
        self.data["generated_answer"] = value

    @property
    def generated_source_documents(self) -> list[Element]:
        """List of documents used to answer this question."""
        return self.data.get("generated_source_documents", [])

    @generated_source_documents.setter
    def generated_source_documents(self, value: list[Element]) -> None:
        """Set the list of used required to answer this question"""
        self.data["generated_source_documents"] = value

    @property
    def metrics(self) -> Optional[dict[str, Any]]:
        """Evaluation metrics."""
        return self.data.get("metrics")

    @metrics.setter
    def metrics(self, value: dict[str, Any]) -> None:
        """Set the evaluation metrics."""
        self.data["metrics"] = value

    @property
    def additional_info(self) -> Optional[dict[str, Any]]:
        """Additional information for specialized datapoints."""
        return self.data.get("additional_info")

    @additional_info.setter
    def additional_info(self, value: Optional[dict[str, Any]]) -> None:
        """Set the additional information for specialized datapoints."""
        self.data["additional_info"] = value

    @property
    def raw(self) -> Optional[Any]:
        """Raw datapoint"""
        return self.data.get("raw")

    @raw.setter
    def raw(self, value: Any) -> None:
        """Set the raw datapoint"""
        self.data["raw"] = value

    @staticmethod
    def deserialize(raw: bytes) -> "EvaluationDataPoint":
        """Deserialize from bytes to a EvaluationDataPoint."""
        from pickle import loads

        return EvaluationDataPoint(loads(raw))


class EvaluationSummary(Document):
    def __init__(
        self,
        document=None,
        **kwargs,
    ):
        super().__init__(document, **kwargs)
        self.data["type"] = "EvaluationSummary"

    @property
    def metrics(self) -> Optional[Any]:
        """Evaluation specific metrics like mrr, accuracy"""
        return self.data.get("metrics")

    @metrics.setter
    def metrics(self, value: Any) -> None:
        self.data["metrics"] = value

    @staticmethod
    def deserialize(raw: bytes) -> "EvaluationSummary":
        """Deserialize from bytes to a EvaluationSummary."""
        from pickle import loads

        return EvaluationSummary(loads(raw))


class EvaluationMetric:
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def metric_name(self) -> str:
        pass

    @abstractmethod
    def evaluate(self, datapoint: EvaluationDataPoint) -> dict[str, Any]:
        pass
