from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SearchRequest(_message.Message):
    __slots__ = ("local_cluster_alias", "absolute_start_millis", "indices", "ccs_minimize_round_trips", "routing", "preference", "source", "search_type", "scroll", "request_cache", "batched_reduce_size", "max_concurrent_shard_requests", "phase_took", "prefilter_size", "pipeline", "indices_options", "final_reduce", "cancel_after_millis")
    class SearchType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        QUERY_THEN_FETCH: _ClassVar[SearchRequest.SearchType]
        DFS_QUERY_THEN_FETCH: _ClassVar[SearchRequest.SearchType]
    QUERY_THEN_FETCH: SearchRequest.SearchType
    DFS_QUERY_THEN_FETCH: SearchRequest.SearchType
    LOCAL_CLUSTER_ALIAS_FIELD_NUMBER: _ClassVar[int]
    ABSOLUTE_START_MILLIS_FIELD_NUMBER: _ClassVar[int]
    INDICES_FIELD_NUMBER: _ClassVar[int]
    CCS_MINIMIZE_ROUND_TRIPS_FIELD_NUMBER: _ClassVar[int]
    ROUTING_FIELD_NUMBER: _ClassVar[int]
    PREFERENCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SEARCH_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCROLL_FIELD_NUMBER: _ClassVar[int]
    REQUEST_CACHE_FIELD_NUMBER: _ClassVar[int]
    BATCHED_REDUCE_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENT_SHARD_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    PHASE_TOOK_FIELD_NUMBER: _ClassVar[int]
    PREFILTER_SIZE_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    INDICES_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    FINAL_REDUCE_FIELD_NUMBER: _ClassVar[int]
    CANCEL_AFTER_MILLIS_FIELD_NUMBER: _ClassVar[int]
    local_cluster_alias: str
    absolute_start_millis: int
    indices: _containers.RepeatedScalarFieldContainer[str]
    ccs_minimize_round_trips: bool
    routing: str
    preference: str
    source: SearchSource
    search_type: SearchRequest.SearchType
    scroll: Scroll
    request_cache: bool
    batched_reduce_size: int
    max_concurrent_shard_requests: int
    phase_took: bool
    prefilter_size: int
    pipeline: str
    indices_options: IndicesOptions
    final_reduce: bool
    cancel_after_millis: int
    def __init__(self, local_cluster_alias: _Optional[str] = ..., absolute_start_millis: _Optional[int] = ..., indices: _Optional[_Iterable[str]] = ..., ccs_minimize_round_trips: bool = ..., routing: _Optional[str] = ..., preference: _Optional[str] = ..., source: _Optional[_Union[SearchSource, _Mapping]] = ..., search_type: _Optional[_Union[SearchRequest.SearchType, str]] = ..., scroll: _Optional[_Union[Scroll, _Mapping]] = ..., request_cache: bool = ..., batched_reduce_size: _Optional[int] = ..., max_concurrent_shard_requests: _Optional[int] = ..., phase_took: bool = ..., prefilter_size: _Optional[int] = ..., pipeline: _Optional[str] = ..., indices_options: _Optional[_Union[IndicesOptions, _Mapping]] = ..., final_reduce: bool = ..., cancel_after_millis: _Optional[int] = ...) -> None: ...

class IndicesOptions(_message.Message):
    __slots__ = ("wildcard_states_open", "wildcard_states_closed", "wildcard_states_hidden", "ignore_unavailable", "ignore_aliases", "allow_no_indices", "forbid_aliases_to_multiple_indices", "forbid_closed_indices", "ignore_throttled")
    WILDCARD_STATES_OPEN_FIELD_NUMBER: _ClassVar[int]
    WILDCARD_STATES_CLOSED_FIELD_NUMBER: _ClassVar[int]
    WILDCARD_STATES_HIDDEN_FIELD_NUMBER: _ClassVar[int]
    IGNORE_UNAVAILABLE_FIELD_NUMBER: _ClassVar[int]
    IGNORE_ALIASES_FIELD_NUMBER: _ClassVar[int]
    ALLOW_NO_INDICES_FIELD_NUMBER: _ClassVar[int]
    FORBID_ALIASES_TO_MULTIPLE_INDICES_FIELD_NUMBER: _ClassVar[int]
    FORBID_CLOSED_INDICES_FIELD_NUMBER: _ClassVar[int]
    IGNORE_THROTTLED_FIELD_NUMBER: _ClassVar[int]
    wildcard_states_open: bool
    wildcard_states_closed: bool
    wildcard_states_hidden: bool
    ignore_unavailable: bool
    ignore_aliases: bool
    allow_no_indices: bool
    forbid_aliases_to_multiple_indices: bool
    forbid_closed_indices: bool
    ignore_throttled: bool
    def __init__(self, wildcard_states_open: bool = ..., wildcard_states_closed: bool = ..., wildcard_states_hidden: bool = ..., ignore_unavailable: bool = ..., ignore_aliases: bool = ..., allow_no_indices: bool = ..., forbid_aliases_to_multiple_indices: bool = ..., forbid_closed_indices: bool = ..., ignore_throttled: bool = ...) -> None: ...

class Scroll(_message.Message):
    __slots__ = ("keep_alive_millis",)
    KEEP_ALIVE_MILLIS_FIELD_NUMBER: _ClassVar[int]
    keep_alive_millis: int
    def __init__(self, keep_alive_millis: _Optional[int] = ...) -> None: ...

class SearchSource(_message.Message):
    __slots__ = ("source_bytes",)
    SOURCE_BYTES_FIELD_NUMBER: _ClassVar[int]
    source_bytes: bytes
    def __init__(self, source_bytes: _Optional[bytes] = ...) -> None: ...
