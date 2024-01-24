import search_request_pb2 as _search_request_pb2
import search_response_pb2 as _search_response_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProcessResponseRequest(_message.Message):
    __slots__ = ("search_response", "search_request")
    SEARCH_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    SEARCH_REQUEST_FIELD_NUMBER: _ClassVar[int]
    search_response: _search_response_pb2.SearchResponse
    search_request: _search_request_pb2.SearchRequest
    def __init__(self, search_response: _Optional[_Union[_search_response_pb2.SearchResponse, _Mapping]] = ..., search_request: _Optional[_Union[_search_request_pb2.SearchRequest, _Mapping]] = ...) -> None: ...

class ProcessResponseResponse(_message.Message):
    __slots__ = ("search_response",)
    SEARCH_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    search_response: _search_response_pb2.SearchResponse
    def __init__(self, search_response: _Optional[_Union[_search_response_pb2.SearchResponse, _Mapping]] = ...) -> None: ...
