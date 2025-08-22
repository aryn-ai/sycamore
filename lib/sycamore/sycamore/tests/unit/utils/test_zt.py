from sycamore.utils.zt import zip_traverse, ZTDict


def test_zt_one_obj():
    raw = {"a": {"b": "c", "d": ["e", "f", "g"], "h": {"i": "j"}}, "k": "l"}
    zraw = ZTDict(raw)
    it = zip_traverse(zraw, order="after")

    seen = []
    seen_k = set()
    for k, v, p in it:
        assert p not in seen
        seen.append(v)
        assert k not in seen_k
        seen_k.add(k)

    seen.clear()
    seen_k.clear()
    it = zip_traverse(zraw, order="before")
    seen.append((raw,))
    for k, v, p in it:
        assert p in seen
        seen.append(v)
        assert k not in seen_k
        seen_k.add(k)


def test_zt_multiple_objects_aligned():
    raw1 = {"a": {"b": "c1", "d": ["e1", "f1"]}, "k": "l1"}
    raw2 = {"a": {"b": "c2", "d": ["e2", "f2"]}, "k": "l2"}

    zraw1 = ZTDict(raw1)
    zraw2 = ZTDict(raw2)

    it = zip_traverse(zraw1, zraw2)

    for k, values, parents in it:
        assert len(values) == 2
        assert len(parents) == 2
        assert all(parents[i][k] == values[i] for i in range(len(values)))


def test_zt_multiple_objects_misaligned_intersect():
    raw1 = {"a": {"b": "c1", "d": "e1"}, "shared": "value1", "only_in_1": "unique1"}
    raw2 = {"a": {"b": "c2", "f": "g2"}, "shared": "value2"}

    zraw1 = ZTDict(raw1)
    zraw2 = ZTDict(raw2)

    it = zip_traverse(zraw1, zraw2, intersect_keys=True)
    key_to_vals = {k: v for k, v, _ in it}

    assert "shared" in key_to_vals
    assert "a" in key_to_vals
    assert "only_in_1" not in key_to_vals
    assert "b" in key_to_vals
    assert "d" not in key_to_vals
    assert "f" not in key_to_vals
    assert not any(None in v for v in key_to_vals.values())


def test_zt_multiple_objects_misaligned_union():
    raw1 = {"a": {"b": "c1", "d": "e1"}, "shared": "v1", "only_in_1": "u1"}
    raw2 = {"a": {"b": "c2", "f": "g2"}, "shared": "v2", "only_in_2": "u2"}

    zraw1 = ZTDict(raw1)
    zraw2 = ZTDict(raw2)

    it = zip_traverse(zraw1, zraw2, intersect_keys=False)
    key_to_vals = {k: v for k, v, _ in it}

    assert key_to_vals["shared"] == ("v1", "v2")
    assert key_to_vals["a"] == ({"b": "c1", "d": "e1"}, {"b": "c2", "f": "g2"})
    assert key_to_vals["only_in_1"] == ("u1", None)
    assert key_to_vals["only_in_2"] == (None, "u2")

    assert key_to_vals["b"] == ("c1", "c2")
    assert key_to_vals["d"] == ("e1", None)
    assert key_to_vals["f"] == (None, "g2")
