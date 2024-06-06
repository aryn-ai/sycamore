from dataclasses import dataclass
from pathlib import Path
import ray
from sycamore.data.document import Document
from sycamore.plan_nodes import Node
from sycamore.writers.base import BaseDBWriter


class FakeClient(BaseDBWriter.client_t):
    def __init__(self, client_params: "FakeClientParams"):
        self.index_name = client_params.index_name
        self.fspath = client_params.fspath

    @classmethod
    def from_client_params(cls, params: "BaseDBWriter.client_params_t") -> "FakeClient":
        assert isinstance(params, FakeClientParams)
        return FakeClient(params)

    def write_many_records(self, records: list["BaseDBWriter.record_t"]):
        for r in records:
            assert isinstance(r, FakeRecord)
            file = self.fspath / self.index_name / r.doc_id
            file.write_text(r.text)

    def create_index_if_missing(self, index_params: "BaseDBWriter.index_params_t"):
        assert isinstance(index_params, FakeIndexParams)
        self.index_name = self.index_name
        (self.fspath / self.index_name).mkdir(exist_ok=True)


class FakeRecord(BaseDBWriter.record_t):
    def __init__(self, doc_id: str, text: str):
        self.doc_id = doc_id
        self.text = text

    @classmethod
    def from_doc(cls, document: Document) -> "FakeRecord":
        return FakeRecord(document.doc_id or "no_id", document.text_representation or "no_text_rep")

    def serialize(self) -> bytes:
        import pickle

        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, byteses: bytes) -> "FakeRecord":
        import pickle

        return pickle.loads(byteses)


class FakeClientParams(BaseDBWriter.client_params_t):
    def __init__(self, path: Path, index_name: str):
        self.fspath = path
        self.index_name = index_name


@dataclass
class FakeIndexParams(BaseDBWriter.index_params_t):
    mode: str


class FakeWriter(BaseDBWriter):
    client_t = FakeClient
    record_t = FakeRecord
    client_params_t = FakeClientParams
    index_params_t = FakeIndexParams


class Common:
    docs = [
        Document({"doc_id": "m1", "text_representation": "it's time to play the music"}),
        Document({"doc_id": "m2", "text_representation": "it's time to light the lights"}),
    ]

    @staticmethod
    def input_node(mocker):
        input_dataset = ray.data.from_items([{"doc": d.serialize()} for d in Common.docs])
        print(input_dataset.take())
        node = mocker.Mock(spec=Node)
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        return node


class TestBaseDBWriter(Common):

    def test_fake_writer_e2e(self, mocker, tmp_path):
        input_node = Common.input_node(mocker)
        client_params = FakeClientParams(tmp_path, "index")
        index_params = FakeIndexParams(mode="filesystem")
        writer = FakeWriter(input_node, client_params, index_params)
        writer.execute()
        index_path: Path = tmp_path / client_params.index_name
        files = list(index_path.iterdir())
        assert len(files) == len(Common.docs)
        assert files[0].name == "m1"
        assert files[0].read_text() == "it's time to play the music"
        assert files[1].name == "m2"
        assert files[1].read_text() == "it's time to light the lights"
