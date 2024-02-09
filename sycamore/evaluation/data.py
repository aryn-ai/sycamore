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
        return self.data.get("query")

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
    def ground_truth_document_url(self) -> Optional[str]:
        """Source document url."""
        return self.data.get("ground_truth_document_url")

    @ground_truth_document_url.setter
    def ground_truth_document_url(self, value: str) -> None:
        """Set the document url."""
        self.data["ground_truth_document_url"] = value

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
    def raw(self) -> Optional[Any]:
        """Raw datapoint"""
        return self.data.get("raw")

    @raw.setter
    def raw(self, value: Any) -> None:
        """Set the raw datapoint"""
        self.data["raw"] = value

    @staticmethod
    def deserialize(raw: bytes) -> "EvaluationDataPoint":
        """Deserialize from bytes to a QADataPoint."""
        from pickle import loads

        return EvaluationDataPoint(loads(raw))
