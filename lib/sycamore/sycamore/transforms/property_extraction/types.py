from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_serializers import field_serializer

from sycamore.data.bbox import BoundingBox
from sycamore.data.element import Element
from sycamore.schema import DataType
from sycamore.llms.prompts.prompts import RenderedPrompt


class AttributionValue(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    element_indices: list[int] = Field(repr=False)
    page: int | list[int] | None = None
    bbox: BoundingBox | None = None
    text_span: tuple[int, int] | None = None
    text_match_score: float | None = None
    text_snippet: str | None = None

    @field_serializer("bbox")
    def serialize_bb(self, bb: BoundingBox | None, _info):
        if bb is None:
            return []
        return [bb.x1, bb.y1, bb.x2, bb.y2]


class RichProperty(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str]
    # TODO: Any -> DataType
    type: Any
    # TODO: Any -> Union[DataType.types]
    value: Any

    is_valid: bool = True

    attribution: AttributionValue | None = None
    # TODO: Any -> Union[DataType.types]
    invalid_guesses: list[Any] = []

    llm_prompt: Optional[RenderedPrompt] = None

    @staticmethod
    def from_prediction(
        prediction: Any, attributable_elements: list[Element], name: Optional[str] = None
    ) -> "RichProperty":
        if isinstance(prediction, dict):
            v_dict: dict[str, RichProperty] = {}
            for k, v in prediction.items():
                v_dict[k] = RichProperty.from_prediction(v, attributable_elements, name=k)
            return RichProperty(
                name=name,
                type=DataType.OBJECT,
                value=v_dict,
            )
        if isinstance(prediction, list):
            v_list: list[RichProperty] = []
            for x in prediction:
                v_list.append(RichProperty.from_prediction(x, attributable_elements))
            return RichProperty(
                name=name,
                type=DataType.ARRAY,
                value=v_list,
            )
        return RichProperty(
            name=name,
            type=type(prediction).__name__,
            value=prediction,
            attribution=AttributionValue(
                element_indices=[e.element_index for e in attributable_elements if e.element_index is not None]
            ),
        )
