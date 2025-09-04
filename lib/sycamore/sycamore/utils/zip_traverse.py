from typing import Protocol, Iterable, Hashable, Any, Literal, Generator


class ZipTraversable(Protocol):
    def keys_zt(self) -> Iterable[Hashable] | None:
        """Return an iterable of keys that can be used to lookup sub-trees via get_zt"""
        ...

    def get_zt(self, key: Hashable) -> "ZipTraversable":
        """Return the subtree corresponding to the key"""
        ...

    def value_zt(self) -> Any:
        """Get the value of this tree node"""
        ...


def zip_traverse(
    *zts: ZipTraversable,
    intersect_keys: bool = False,
    order: Literal["before", "after"] = "after",
) -> Generator[tuple[Hashable, tuple[Any, ...], tuple[Any, ...]], None, None]:
    """
    Traverse zts in sync, i.e. assuming each tree has the same structure,
    yield values from the corresponding locations in each tree together.

    Args:
        zts: ZipTraversable trees
        intersect_keys: true -> only yield values that are in all trees.
                        false -> yield values in any tree.
        order: traversal order (before = top-down, after = bottom-up)

    Yields:
        tuple: key, values, parents
            values and parents are tuples of objects from the trees in the
            order they were supplied.
    """
    # Figure out keys. A tree may report None for keys_zt to say "I don't care
    # about the keys", in which case, we drop them from the set operation entirely
    # unless only Nones exist, in which case we set the keys to exactly {None}.
    # The only place this behavior is used (to my knowledge) is schema::ArrayProperty
    key_iters = [zt.keys_zt() for zt in zts]
    if all(ki is None for ki in key_iters):
        key_sets: list[set[None | Hashable]] = [{None}]
    else:
        key_sets = [set(ki) for ki in key_iters if ki is not None]
        if all(len(ks) == 0 for ks in key_sets) and None in key_iters:
            key_sets.append({None})
    if intersect_keys:
        keys = set.intersection(*key_sets)
    else:
        keys = set.union(*key_sets)

    for k in sorted(keys):  # type: ignore  # Hashable is not necessarily sortable, but strings and ints are
        if order == "before":
            yield (k, tuple(zt.get_zt(k).value_zt() for zt in zts), tuple(zt.value_zt() for zt in zts))
        yield from zip_traverse(*(zt.get_zt(k) for zt in zts), intersect_keys=intersect_keys, order=order)
        if order == "after":
            yield (k, tuple(zt.get_zt(k).value_zt() for zt in zts), tuple(zt.value_zt() for zt in zts))


class ZTDict(dict, ZipTraversable):
    def keys_zt(self) -> Iterable[Hashable] | None:
        return self.keys()

    def get_zt(self, key: Hashable) -> ZipTraversable:
        x = self.get(key)
        if isinstance(x, dict):
            return ZTDict(x)
        if isinstance(x, list):
            return ZTList(x)
        return ZTLeaf(x)

    def value_zt(self) -> Any:
        return self


class ZTList(list, ZipTraversable):
    def keys_zt(self) -> Iterable[Hashable] | None:
        return range(len(self))

    def get_zt(self, key: Hashable) -> ZipTraversable:
        if not isinstance(key, int):
            return ZTLeaf(None)
        if key >= len(self) or key < 0:
            return ZTLeaf(None)
        x = self[key]
        if isinstance(x, dict):
            return ZTDict(x)
        if isinstance(x, list):
            return ZTList(x)
        return ZTLeaf(x)

    def value_zt(self) -> Any:
        return self


class ZTLeaf:
    def __init__(self, value):
        self.value = value

    def keys_zt(self) -> Iterable[Hashable] | None:
        return ()

    def get_zt(self, key: Hashable) -> ZipTraversable:
        return ZTLeaf(None)

    def value_zt(self) -> Any:
        return self.value
