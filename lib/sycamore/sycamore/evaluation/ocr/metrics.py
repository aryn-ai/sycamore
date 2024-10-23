from abc import ABC, abstractmethod
from ray.data.aggregate import AggregateFn
from jiwer import mer, wer, cer, wil
from sycamore.data import Document, MetadataDocument
from typing import Optional, Union, Callable
from sycamore.evaluation.ocr.data import OCREvalDocument


class OCRMetric(ABC):

    @abstractmethod
    def score(self, source_string: str, predicted_string: str) -> float:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def to_aggregate_fn(self) -> AggregateFn:
        def init(k):
            return 0.0, 0.0

        def acc_row(agg, row):
            ed = Document.deserialize(row["doc"])
            if isinstance(ed, MetadataDocument):
                return agg
            ed = OCREvalDocument(ed)
            score = ed.metrics.get(self.get_name(), 0)
            return agg[0] + score, agg[1] + 1.0

        def merge(agg1, agg2):
            return agg1[0] + agg2[0], agg1[1] + agg2[1]

        def finalize(agg):
            return agg[0] / agg[1]

        return AggregateFn(init=init, name=self.get_name(), merge=merge, accumulate_row=acc_row, finalize=finalize)


def apply_metric(metric: "OCRMetric") -> Callable[[Document], Document]:
    def f(doc: Document):
        ed = OCREvalDocument(doc.data)
        if ed.pred_text is None or ed.gt_text is None:
            score = 0.0
        else:
            score = metric.score(ed.gt_text, ed.pred_text)
        ed.metrics[metric.get_name()] = score
        return ed

    f.__name__ = metric.get_name()
    return f


class CharacterErrorRate(OCRMetric):
    def score(
        self, source_string: str, predicted_string: str, special_characters: Optional[Union[list[str], set[str]]] = None
    ) -> float:
        if special_characters:
            special_set = set(special_characters) if not isinstance(special_characters, set) else special_characters
            source_string_special = "".join(char for char in source_string if char in special_set)
            predicted_string_special = "".join(char for char in predicted_string if char in special_set)
            return cer(source_string_special, predicted_string_special)
        return cer(source_string, predicted_string)

    def get_name(self) -> str:
        return "Character Error Rate"


class WordErrorRate(OCRMetric):
    def score(self, source_string: str, predicted_string: str) -> float:
        return wer(source_string, predicted_string)

    def get_name(self) -> str:
        return "Word Error Rate"


class MatchErrorRate(OCRMetric):
    def score(self, source_string: str, predicted_string: str) -> float:
        return mer(source_string, predicted_string)

    def get_name(self) -> str:
        return "Match Error Rate"


class WordInformationLost(OCRMetric):
    def score(self, source_string: str, predicted_string: str) -> float:
        return wil(source_string, predicted_string)

    def get_name(self) -> str:
        return "Word Information Lost"
