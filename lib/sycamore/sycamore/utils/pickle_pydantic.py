import io
from typing import Any, Dict, Optional, Type
from ray import cloudpickle
from pydantic import BaseModel


def safe_cloudpickle(obj: Any) -> bytes:
    model_namespaces = {}

    with io.BytesIO() as f:
        pickler = cloudpickle.CloudPickler(f)

        for ModelClass in BaseModel.__subclasses__():
            model_namespaces[ModelClass] = ModelClass.__pydantic_parent_namespace__
            ModelClass.__pydantic_parent_namespace__ = None

        try:
            pickler.dump(obj)
            return f.getvalue()
        finally:
            for ModelClass, namespace in model_namespaces.items():
                ModelClass.__pydantic_parent_namespace__ = namespace


def safe_cloudunpickle(pickled_obj: bytes) -> Any:
    model_namespaces: Dict[Type[BaseModel], Optional[Dict[str, Any]]] = {}

    # Collect the current parent namespaces before unpickling
    for ModelClass in BaseModel.__subclasses__():
        model_namespaces[ModelClass] = getattr(ModelClass, "__pydantic_parent_namespace__", None)
        ModelClass.__pydantic_parent_namespace__ = None

    try:
        with io.BytesIO(pickled_obj) as f:
            obj = cloudpickle.load(f)
            return obj
    finally:
        # Restore the original __pydantic_parent_namespace__ attributes
        for ModelClass, namespace in model_namespaces.items():
            ModelClass.__pydantic_parent_namespace__ = namespace
