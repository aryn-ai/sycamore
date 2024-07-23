from dataclasses import dataclass
from sycamore.data.document import Document
from sycamore.connectors.base_reader import BaseDBReader
from typing import Any
import pytest


class FakeClient(BaseDBReader.Client):
    def __init__(self, client_params: "FakeClientParams"):
        pass

    @classmethod
    def from_client_params(cls, params: BaseDBReader.ClientParams) -> "FakeClient":
        assert isinstance(params, FakeClientParams)
        return FakeClient(params)

    def read_records(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, FakeQueryParams)
        record = Common.record
        return record

    def check_target_presence(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(query_params, FakeQueryParams)
        return query_params.target_name == "target"


class FakeQueryResponse(BaseDBReader.QueryResponse):
    def __init__(self, output: list[Any]):
        self.output = output

    def to_docs(self, query_params: BaseDBReader.QueryParams):
        assert isinstance(self, FakeQueryResponse) and isinstance(query_params, FakeQueryParams)
        docs = []
        for r in self.output:
            docs.append(Document(r))
        return docs


@dataclass
class FakeClientParams(BaseDBReader.ClientParams):
    pass


@dataclass
class FakeQueryParams(BaseDBReader.QueryParams):
    target_name: str


class FakeReader(BaseDBReader):
    Client = FakeClient
    QueryResponse = FakeQueryResponse
    ClientParams = FakeClientParams
    QueryParams = FakeQueryParams


class Common:
    record = FakeQueryResponse(
        [
            {"doc_id": "m1", "text_representation": "it's time to play the music"},
            {"doc_id": "m2", "text_representation": "it's time to light the lights"},
        ]
    )
    docs = [
        Document({"doc_id": "m1", "text_representation": "it's time to play the music"}),
        Document({"doc_id": "m2", "text_representation": "it's time to light the lights"}),
    ]


class TestBaseDBReader(Common):

    def test_fake_reader_e2e_happy(self):
        client_params = FakeClientParams()
        target_params = FakeQueryParams(target_name="target")
        reader = FakeReader(client_params, target_params)
        read_output = reader.read_docs()
        assert len(read_output) == len(Common.record.output)
        assert read_output[0].doc_id == "m1"
        assert read_output[0].text_representation == "it's time to play the music"
        assert read_output[1].doc_id == "m2"
        assert read_output[1].text_representation == "it's time to light the lights"

    def test_fake_reader_has_correct_inner_classes(self):
        assert FakeReader.Client == FakeClient
        assert FakeReader.ClientParams == FakeClientParams
        assert FakeReader.QueryResponse == FakeQueryResponse
        assert FakeReader.QueryParams == FakeQueryParams

    def test_nonmatching_target_params_then_fail(self):
        client_params = FakeClientParams()
        target_params = FakeQueryParams(target_name="notthetarget")
        reader = FakeReader(client_params, target_params)
        with pytest.raises(ValueError) as einfo:
            reader.read_docs()
        assert "target_name='notthetarget'" in str(einfo.value)
