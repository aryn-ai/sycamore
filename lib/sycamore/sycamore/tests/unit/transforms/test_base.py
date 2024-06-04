import ray

from ray.data import ActorPoolStrategy

from sycamore.data import Document, MetadataDocument
from sycamore.plan_nodes import Node
from sycamore.transforms.base import BaseMapTransform, CompositeTransform


class Common:
    dicts = [
        {"doc_id": "pb1", "doc": "Beat it or I'll call the Brute Squad."},
        {"doc_id": "pb2", "doc": "I'm on the Brute Squad."},
        {"doc_id": "pb3", "doc": "You ARE the Brute Squad!"},
    ]
    ndocs = len(dicts)

    @staticmethod
    def input_node(mocker):
        input_dataset = ray.data.from_items([{"doc": Document(d).serialize()} for d in TestBaseMapTransform.dicts])
        node = mocker.Mock(spec=Node)
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset

        return node

    @staticmethod
    def outputs(node: Node):
        all_docs = [Document.from_row(r) for r in node.execute().take()]
        docs = [d for d in all_docs if not isinstance(d, MetadataDocument)]
        metadata = [d for d in all_docs if isinstance(d, MetadataDocument)]
        return (docs, metadata)


class TestBaseMapTransform(Common):
    @staticmethod
    def fn_a(docs: list[Document], arg: str, *, extra2="unset") -> list[Document]:
        ret: list[Document] = []
        for d in docs:
            if "trace" not in d.properties:
                d.properties["trace"] = []
            assert d.properties["trace"] is not None  # make mypy happy
            print(f"During processing {d.doc_id} = {d.lineage_id}")
            d.properties["trace"].append(["fnA", arg, d.lineage_id])
            ret.append(d)

            assert d.doc_id is not None
            m1 = MetadataDocument(id=d.doc_id + "#1", trace=d.properties["trace"].copy())
            m2 = MetadataDocument(id=d.doc_id + "#2", extra=arg, extra2=extra2)
            ret.extend([m1, m2])

        return ret

    def test_simple(self, mocker) -> None:
        (docs, mds) = self.outputs(
            BaseMapTransform(
                self.input_node(mocker),
                f=self.fn_a,
                args=["simple"],
                kwargs={"extra2": "kwarg"},
                enable_auto_metadata=True,
            )
        )

        ndocs = self.ndocs

        assert len(docs) == ndocs
        assert len(mds) == ndocs * 3  # 3 auto-generated lineage links + 6 manually constructed metadata

        # Check that all the docs made it through
        docs.sort(key=lambda x: x.doc_id)
        id_to_num = {}
        for i in range(ndocs):
            assert docs[i].doc_id == f"pb{i+1}"
            id_to_num[docs[i].lineage_id] = i

        # Check that they got the expected properties
        for d in docs:
            t = d.properties["trace"]
            assert len(t) == 1
            (fn, arg, lineage) = t[0]
            assert fn == "fnA"
            assert arg == "simple"
            assert isinstance(lineage, str)
            assert lineage != d.lineage_id
            id_to_num[lineage] = id_to_num[d.lineage_id]

        # Check that the manually created metadata made it through
        manual_mds = [m for m in mds if "id" in m.metadata]
        assert len(manual_mds) == 2 * ndocs
        manual_mds.sort(key=lambda x: x.metadata["id"])

        for i in range(ndocs):
            assert manual_mds[2 * i].metadata["trace"] == docs[i].properties["trace"]
            assert manual_mds[2 * i + 1].metadata["extra"] == "simple"
            assert manual_mds[2 * i + 1].metadata["extra2"] == "kwarg"

        # Check that the automatically generated metadata is correct.
        auto_mds = [m for m in mds if "lineage_links" in m.metadata]
        assert len(auto_mds) == ndocs
        for m in auto_mds:
            ll = m.metadata["lineage_links"]
            from_ids = ll["from_ids"]
            to_ids = ll["to_ids"]
            assert len(from_ids) == 1
            assert len(to_ids) == 1
            assert from_ids[0] != to_ids[0]
            assert id_to_num[from_ids[0]] == id_to_num[to_ids[0]]

    def test_passthrough(self, mocker) -> None:
        a = BaseMapTransform(self.input_node(mocker), f=self.fn_a, args=["simple"], enable_auto_metadata=True)
        b = BaseMapTransform(a, f=lambda x: x, enable_auto_metadata=True)
        (docs, mds) = self.outputs(b)
        ndocs = self.ndocs

        assert len(docs) == ndocs
        assert len(mds) == ndocs * 4  # 3 auto-generated lineage links + 6 manually constructed metadata

        # Check that all the docs made it through
        docs.sort(key=lambda x: x.doc_id)
        id_to_num = {}
        for i in range(ndocs):
            assert docs[i].doc_id == f"pb{i+1}"
            id_to_num[docs[i].lineage_id] = i

        # Check that they got the expected properties
        for d in docs:
            t = d.properties["trace"]
            assert len(t) == 1
            (fn, arg, lineage) = t[0]
            assert fn == "fnA"
            assert arg == "simple"
            assert isinstance(lineage, str)
            assert lineage != d.lineage_id
            id_to_num[lineage] = id_to_num[d.lineage_id]

        # Check that the manually created metadata made it through
        manual_mds = [m for m in mds if "id" in m.metadata]
        assert len(manual_mds) == 2 * ndocs
        manual_mds.sort(key=lambda x: x.metadata["id"])

        for i in range(ndocs):
            assert manual_mds[2 * i].metadata["trace"] == docs[i].properties["trace"]
            assert manual_mds[2 * i + 1].metadata["extra"] == "simple"

        # Check that the automatically generated metadata is correct.
        auto_mds = [m for m in mds if "lineage_links" in m.metadata]
        assert len(auto_mds) == ndocs * 2
        notfound = 0
        for m in auto_mds:
            ll = m.metadata["lineage_links"]
            from_ids = ll["from_ids"]
            to_ids = ll["to_ids"]
            assert len(from_ids) == 1
            assert len(to_ids) == 1
            assert from_ids[0] != to_ids[0]
            if from_ids[0] not in id_to_num:
                notfound = notfound + 1
                id_to_num[from_ids[0]] = id_to_num[to_ids[0]]
            elif to_ids[0] not in id_to_num:
                notfound = notfound + 1
                id_to_num[to_ids[0]] = id_to_num[from_ids[0]]

            assert id_to_num[from_ids[0]] == id_to_num[to_ids[0]]

        assert notfound == 3

    def test_class(self, mocker) -> None:
        class Test:
            def __init__(self, a, *, b="unset"):
                self.a = a
                self.b = b
                self.c = 0

            def __call__(self, docs: list[Document], e, *, f="unset"):
                ret = docs.copy()
                for d in docs:
                    ret.append(MetadataDocument(id=d.doc_id, lid=d.lineage_id, a=self.a, b=self.b, c=self.c, e=e, f=f))
                    self.c = self.c + 1

                return ret

        (docs, mds) = self.outputs(
            BaseMapTransform(
                self.input_node(mocker),
                f=Test,
                constructor_args=["c1"],
                constructor_kwargs={"b": "c2"},
                args=["a1"],
                kwargs={"f": "a2"},
                enable_auto_metadata=True,
            )
        )

        ndocs = self.ndocs
        assert len(docs) == ndocs
        assert len(mds) == ndocs * 2

        lid_to_did = {}
        for d in docs:
            lid_to_did[d.lineage_id] = d.doc_id

        for d in mds:
            md = d.metadata
            if "lid" in md:
                lid_to_did[md["lid"]] = md["id"]

        lineage = 0
        custom = 0
        c = 0
        for d in mds:
            md = d.metadata
            if "lineage_links" in md:
                lineage = lineage + 1
                ll = md["lineage_links"]
                from_ids = ll["from_ids"]
                to_ids = ll["to_ids"]
                assert len(from_ids) == 1
                assert len(to_ids) == 1
                assert from_ids[0] != to_ids[0]
                assert lid_to_did[from_ids[0]] == lid_to_did[to_ids[0]]
            else:
                custom = custom + 1
                assert lid_to_did[md["lid"]] == md["id"]
                assert md["a"] == "c1"
                assert md["b"] == "c2"
                # relies on ray preserving order, verifies we make a single instance of the class.
                assert md["c"] == c
                c = c + 1
                assert md["e"] == "a1"
                assert md["f"] == "a2"

    def test_object(self, mocker) -> None:
        class Test:
            def __init__(self, oid):
                self.oid = oid
                self.c = 0

            def __call__(self, docs: list[Document]):
                ret = docs.copy()
                for d in docs:
                    ret.append(MetadataDocument(oid=self.oid, c=self.c, n=len(docs)))
                    self.c = self.c + 1

                return ret

        as_function = BaseMapTransform(
            self.input_node(mocker),
            f=Test("as_function"),
            enable_auto_metadata=False,
        )
        as_object = BaseMapTransform(
            as_function, f=Test("as_object"), compute=ActorPoolStrategy(size=1), enable_auto_metadata=False
        )

        (docs, mds) = self.outputs(as_object)

        ndocs = self.ndocs
        assert len(mds) == ndocs * 2

        md_fn = [m.metadata for m in mds if m.metadata["oid"] == "as_function"]
        md_obj = [m.metadata for m in mds if m.metadata["oid"] == "as_object"]

        assert len(md_fn) == ndocs
        assert len(md_obj) == ndocs

        # Unexpectedly ray is fusing the two steps together resulting in the object state being
        # preserved.  Testing the actual behavior rather than the expected behavior of each call to
        # as_function getting a separate instance. It is unclear whether this behavior is
        # guaranteed or merely an artifact of the test.

        c = 0
        md_fn.sort(key=lambda m: m["c"])
        print(f"md_fn: {md_fn}")
        for m in md_fn:
            # each one should get a new instance
            assert m["c"] == c
            assert m["n"] == 1
            c = c + 1

        c = 0
        md_obj.sort(key=lambda m: m["c"])
        print(f"md_obj: {md_obj}")
        for m in md_obj:
            assert m["c"] == c
            assert m["n"] == 1
            c = c + 1


class TestCompositeTransform(Common):
    def test_simple(self, mocker) -> None:
        start = TestBaseMapTransform.input_node(mocker)

        def fn(docs: list[Document], arg: str) -> list[Document]:
            for d in docs:
                if "val" not in d.properties:
                    d.properties["val"] = []
                d.properties["val"].append(arg)

            return docs

        last = CompositeTransform(start, [{"f": fn, "args": [1]}, {"f": fn, "args": [3]}, {"f": fn, "args": [2]}])

        def simple_check(docs):
            assert len(docs) == Common.ndocs
            for d in docs:
                assert "val" in d.properties
                v = d.properties["val"]
                assert len(v) == 3
                assert v[0] == 1
                assert v[1] == 3
                assert v[2] == 2

        docs = last._local_process([Document(d) for d in Common.dicts])
        simple_check(docs)

        (docs, mds) = self.outputs(last)
        simple_check(docs)
