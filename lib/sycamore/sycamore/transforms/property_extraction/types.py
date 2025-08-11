from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_serializers import field_serializer
from pydantic.functional_validators import field_validator

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

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bb(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            assert len(value) == 4
            return BoundingBox(*value)
        return value


class RichProperty(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str]

    type: DataType
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
            type=DataType.from_python(prediction) if prediction is not None else DataType.STRING,
            value=prediction,
            attribution=AttributionValue(
                element_indices=[e.element_index for e in attributable_elements if e.element_index is not None]
            ),
        )

    def to_python(self):
        if self.type == DataType.ARRAY:
            assert isinstance(self.value, list)
            return [v.to_python() for v in self.value]
        if self.type == DataType.OBJECT:
            assert isinstance(self.value, dict)
            return {k: v.to_python() for k, v in self.value.items()}
        return self.value

    @classmethod
    def validate_recursive(cls, obj: Any) -> "RichProperty":
        v = cls.model_validate(obj)
        if isinstance(v.value, list):
            for i, x in enumerate(v.value):
                v.value[i] = cls.validate_recursive(x)
        elif isinstance(v.value, dict):
            for k, x in v.value.items():
                v.value[k] = cls.validate_recursive(x)
        return v

    def dump_recursive(self) -> Any:
        v = self.model_dump()
        if isinstance(v["value"], list):
            for i, x in enumerate(v["value"]):
                v["value"][i] = x.dump_recursive()
        elif isinstance(v["value"], dict):
            for k, x in v["value"].items():
                v["value"][k] = x.dump_recursive()
        return v
