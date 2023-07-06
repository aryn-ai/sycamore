from typing import (Any, Callable, Dict, Iterable, List, Optional, Type)

from pyarrow import Table
from ray.data import (ActorPoolStrategy, Dataset)

from sycamore.execution import (Node, UnaryNode)
from sycamore.data import Document


def generate_map_function(f: Callable[[Document], Document]) -> Callable[
    [Dict[str, Any]], Dict[str, Any]
]:
    def ray_callable(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        document = f(Document(input_dict))
        return document.data

    return ray_callable


def generate_map_class(c: Type[Callable[[Document], Document]]) -> Type[
    Callable[[Dict[str, Any]], Dict[str, Any]]
]:
    def ray_init(self):
        self.base = c()

    def ray_callable(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        document = self.base(Document(input_dict))
        return document.data

    new_class = type(
        "CustomRay" + c.__name__,
        (),
        {"__init__": ray_init, "__call__": ray_callable})
    return new_class


def generate_flat_map_function(
        f: Callable[[Document], List[Document]]) -> Callable[
    [Dict[str, Any]], List[Dict[str, Any]]
]:
    def ray_callable(input_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents = f(Document(input_dict))
        return [document.data for document in documents]

    return ray_callable


def generate_flat_map_class(c: Type[Callable[[Document], List[Document]]]) ->\
        Type[Callable[[Dict[str, Any]], List[Dict[str, Any]]]]:

    def ray_init(self):
        self.base = c()

    def ray_callable(self, input_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents = self.base(Document(input_dict))
        return [document.data for document in documents]

    new_class = type(
        "CustomRay" + c.__name__,
        (),
        {"__init__": ray_init, "__call__": ray_callable})
    return new_class


def generate_map_batch_function(
        f: Callable[[List[Document]], List[Document]]) -> Callable[
    [Table], Table
]:
    def ray_callable(input_table: Table) -> Table:
        input_docs = [Document(t) for t in input_table.to_pylist()]
        output_docs = f(input_docs)
        output_dicts = [doc.data for doc in output_docs]
        from pandas import DataFrame
        df = DataFrame(output_dicts)
        output_table = Table.from_pandas(df)
        return output_table

    return ray_callable


def generate_map_batch_class(
        c: Type[Callable[[List[Document]], List[Document]]],
        f_args: Optional[Iterable[Any]] = None,
        f_kwargs: Optional[Dict[str, Any]] = None,
        f_constructor_args: Optional[Iterable[Any]] = None,
        f_constructor_kwargs: Optional[Dict[str, Any]] = None) -> Type[
    Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
]:
    if f_constructor_args is None:
        f_constructor_args = tuple()
    if f_constructor_kwargs is None:
        f_constructor_kwargs = {}
    if f_args is None:
        f_args = tuple()
    if f_kwargs is None:
        f_kwargs = {}

    def ray_init(self):
        self.base = c(*f_constructor_args, **f_constructor_kwargs)

    def ray_callable(self, input_table: Table) -> Table:
        input_docs = [Document(t) for t in input_table.to_pylist()]
        output_docs = self.base(input_docs, *f_args, **f_kwargs)
        output_dicts = [doc.data for doc in output_docs]
        from pandas import DataFrame
        df = DataFrame(output_dicts)
        output_table = Table.from_pandas(df)
        return output_table

    new_class = type(
        "CustomRay" + c.__name__,
        (),
        {"__init__": ray_init, "__call__": ray_callable})
    return new_class


class Map(UnaryNode):
    def __init__(
            self,
            child: Node,
            *,
            f: Callable[[Document], Document],
            **resource_args):
        super().__init__(child, **resource_args)
        self._f = f

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        if isinstance(self._f, type):
            ray_callable = generate_map_class(self._f)
            return input_dataset.map(
                ray_callable,
                compute=ActorPoolStrategy(size=1),
                **self.resource_args)
        else:
            ray_callable = generate_map_function(self._f)
            return input_dataset.map(ray_callable, **self.resource_args)


class FlatMap(UnaryNode):
    def __init__(
            self,
            child: Node,
            *,
            f: Callable[[Document], List[Document]],
            **resource_args):
        super().__init__(child, **resource_args)
        self._f = f

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        if isinstance(self._f, type):
            ray_callable = generate_flat_map_class(self._f)
            return input_dataset.flat_map(
                ray_callable,
                compute=ActorPoolStrategy(size=1),
                **self.resource_args)
        else:
            ray_callable = generate_flat_map_function(self._f)
            return input_dataset.flat_map(
                ray_callable, **self.resource_args)


class MapBatch(UnaryNode):
    def __init__(
            self,
            child: Node,
            *,
            f: Callable[[List[Document]], List[Document]],
            f_args: Optional[Iterable[Any]] = None,
            f_kwargs: Optional[Dict[str, Any]] = None,
            f_constructor_args: Optional[Iterable[Any]] = None,
            f_constructor_kwargs: Optional[Dict[str, Any]] = None,
            **resource_args):
        super().__init__(child, **resource_args)
        self._f = f
        self._f_args = f_args
        self._f_kwargs = f_kwargs
        self._f_constructor_args = f_constructor_args
        self._f_constructor_kwargs = f_constructor_kwargs

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        if isinstance(self._f, type):
            ray_callable = generate_map_batch_class(
                self._f, self._f_args, self._f_kwargs,
                self._f_constructor_args, self._f_constructor_kwargs)
            return input_dataset.map_batches(
                ray_callable,
                compute=ActorPoolStrategy(size=1),
                batch_format="pyarrow",
                **self.resource_args)
        else:
            ray_callable = generate_map_batch_function(self._f)
            return input_dataset.map_batches(
                ray_callable, batch_format="pyarrow", **self.resource_args)
