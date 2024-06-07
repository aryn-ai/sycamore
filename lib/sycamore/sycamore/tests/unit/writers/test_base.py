from dataclasses import dataclass
from pathlib import Path
from sycamore.data.document import Document, MetadataDocument
from sycamore.plan_nodes import Node
from sycamore.writers.base import BaseDBWriter
from typing import Optional


class FakeClient(BaseDBWriter.Client):
    def __init__(self, client_params: "FakeClientParams"):
        self.fspath = client_params.fspath

    @classmethod
    def from_client_params(cls, params: "BaseDBWriter.ClientParams") -> "FakeClient":
        assert isinstance(params, FakeClientParams)
        return FakeClient(params)

    def write_many_records(self, records: list["BaseDBWriter.Record"], target_params: "BaseDBWriter.TargetParams"):
        for r in records:
            assert isinstance(r, FakeRecord) and isinstance(target_params, FakeTargetParams)
            file = self.fspath / target_params.dirname / r.doc_id
            file.write_text(r.text)

    def create_index_idempotent(self, target_params: "BaseDBWriter.TargetParams"):
        assert isinstance(target_params, FakeTargetParams)
        (self.fspath / target_params.dirname).mkdir(exist_ok=True)

    def get_existing_target_params(
        self, target_params: "BaseDBWriter.TargetParams"
    ) -> Optional["BaseDBWriter.TargetParams"]:
        assert isinstance(target_params, FakeTargetParams)
        if (self.fspath / target_params.dirname).exists():
            return FakeTargetParams(dirname=target_params.dirname)
        return None


class FakeRecord(BaseDBWriter.Record):
    def __init__(self, doc_id: str, text: str):
        self.doc_id = doc_id
        self.text = text

    @classmethod
    def from_doc(cls, document: Document) -> "FakeRecord":
        assert not isinstance(document, MetadataDocument)
        return FakeRecord(document.doc_id or "no_id", document.text_representation or "no_text_rep")


@dataclass
class FakeClientParams(BaseDBWriter.ClientParams):
    fspath: Path


@dataclass
class FakeTargetParams(BaseDBWriter.TargetParams):
    dirname: str


class FakeWriter(BaseDBWriter):
    Client = FakeClient
    Record = FakeRecord
    ClientParams = FakeClientParams
    TargetParams = FakeTargetParams


class Common:
    docs = [
        Document({"doc_id": "m1", "text_representation": "it's time to play the music"}),
        Document({"doc_id": "m2", "text_representation": "it's time to light the lights"}),
    ]


class TestBaseDBWriter(Common):

    def test_fake_writer_e2e_happy(self, mocker, tmp_path):
        input_node = mocker.Mock(spec=Node)
        client_params = FakeClientParams(fspath=tmp_path)
        target_params = FakeTargetParams(dirname="target")
        writer = FakeWriter(input_node, client_params, target_params)
        writer.write_docs(Common.docs)
        index_path: Path = tmp_path / target_params.dirname
        files = list(index_path.iterdir())
        assert len(files) == len([d for d in Common.docs if not isinstance(d, MetadataDocument)])
        assert files[0].name == "m1"
        assert files[0].read_text() == "it's time to play the music"
        assert files[1].name == "m2"
        assert files[1].read_text() == "it's time to light the lights"

    def test_fake_writer_has_correct_inner_classes(self):
        assert FakeWriter.Client == FakeClient
        assert FakeWriter.ClientParams == FakeClientParams
        assert FakeWriter.Record == FakeRecord
        assert FakeWriter.TargetParams == FakeTargetParams
