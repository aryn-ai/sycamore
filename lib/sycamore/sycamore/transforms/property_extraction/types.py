from typing import Optional, Any, Iterable, Hashable
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_serializers import field_serializer
from pydantic.functional_validators import field_validator

from sycamore.data.bbox import BoundingBox
from sycamore.schema import DataType
from sycamore.llms.prompts.prompts import RenderedPrompt
from sycamore.utils.zip_traverse import ZTDict, ZTLeaf, ZipTraversable, zip_traverse


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

    def keys_zt(self) -> Iterable[Hashable] | None:
        if self.type is DataType.OBJECT:
            assert isinstance(self.value, dict)
            return self.value.keys()
        if self.type is DataType.ARRAY:
            assert isinstance(self.value, list)
            return range(len(self.value))
        return ()

    def get_zt(self, key: Hashable) -> "ZipTraversable":
        if key is None:
            return ZTLeaf(None)
        if self.type is DataType.OBJECT:
            assert isinstance(self.value, dict)
            v = self.value.get(key, ZTLeaf(None))
            assert isinstance(v, (RichProperty, ZTLeaf))
            return v
        if self.type is DataType.ARRAY:
            assert isinstance(self.value, list)
            assert isinstance(key, int)
            if key >= len(self.value) or key < 0:
                return ZTLeaf(None)
            v = self.value[key]
            assert isinstance(v, RichProperty)
            return v
        return ZTLeaf(None)

    def value_zt(self) -> Any:
        return self

    def _add_subprop(self, other: "RichProperty"):
        if other.name is None:
            assert self.type is DataType.ARRAY
            self.value.append(other)
        else:
            assert self.type is DataType.OBJECT
            self.value[other.name] = other

    @staticmethod
    def from_prediction(prediction: dict[str, Any]) -> "RichProperty":
        res = RichProperty(name=None, value={}, type=DataType.OBJECT)
        ztp = ZTDict(prediction)
        for k, (pred_v, res_v), (pred_p, res_p) in zip_traverse(ztp, res, order="before", intersect_keys=False):
            name = k if isinstance(k, str) else None
            if pred_v is None:
                # Might want to do something more intelligent here
                # but would need a reference to the schema to determine
                # the type
                continue
            dt = DataType.from_python(pred_v)
            new_rp = RichProperty(name=name, type=dt, value=pred_v)
            if dt is DataType.OBJECT:
                new_rp.value = {}
            if dt is DataType.ARRAY:
                new_rp.value = []
            res_p._add_subprop(new_rp)
        return res

    def to_python(self):
        # I tried writing this with ziptraverse but it just wasn't as clean
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
