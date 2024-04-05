import ray

from sycamore.data import Document, MetadataDocument
from sycamore.plan_nodes import Node
from sycamore.transforms.base import BaseMap


class TestBaseMap:
    dicts = [
        {"doc_id": "pb1", "doc": "Beat it or I'll call the Brute Squad."},
        {"doc_id": "pb2", "doc": "I'm on the Brute Squad."},
        {"doc_id": "pb3", "doc": "You ARE the Brute Squad!"},
    ]
    ndocs = len(dicts)

    @staticmethod
    def input_node(mocker):
        input_dataset = ray.data.from_items([{"doc": Document(d).serialize()} for d in TestBaseMap.dicts])
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

    def test_simple(self, mocker):
        (docs, mds) = self.outputs(
            BaseMap(self.input_node(mocker), f=self.fn_a, args=["simple"], kwargs={"extra2": "kwarg"})
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

    def test_passthrough(self, mocker):
        a = BaseMap(self.input_node(mocker), f=self.fn_a, args=["simple"])
        b = BaseMap(a, f=lambda x: x)
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

    def test_class(self, mocker):
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
            BaseMap(
                self.input_node(mocker),
                f=Test,
                constructor_args=["c1"],
                constructor_kwargs={"b": "c2"},
                args=["a1"],
                kwargs={"f": "a2"},
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
                # relies on ray preserving order
                assert md["c"] == c
                c = c + 1
                assert md["e"] == "a1"
                assert md["f"] == "a2"
