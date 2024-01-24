from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SearchResponse(_message.Message):
    __slots__ = ("internal_response", "scroll_id", "point_in_time_id", "total_shards", "successful_shards", "skipped_shards", "shard_failures", "clusters", "took_in_millis", "phase_took")
    INTERNAL_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    SCROLL_ID_FIELD_NUMBER: _ClassVar[int]
    POINT_IN_TIME_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SHARDS_FIELD_NUMBER: _ClassVar[int]
    SUCCESSFUL_SHARDS_FIELD_NUMBER: _ClassVar[int]
    SKIPPED_SHARDS_FIELD_NUMBER: _ClassVar[int]
    SHARD_FAILURES_FIELD_NUMBER: _ClassVar[int]
    CLUSTERS_FIELD_NUMBER: _ClassVar[int]
    TOOK_IN_MILLIS_FIELD_NUMBER: _ClassVar[int]
    PHASE_TOOK_FIELD_NUMBER: _ClassVar[int]
    internal_response: SearchResponseSections
    scroll_id: str
    point_in_time_id: str
    total_shards: int
    successful_shards: int
    skipped_shards: int
    shard_failures: _containers.RepeatedCompositeFieldContainer[SearchShardFailure]
    clusters: Clusters
    took_in_millis: int
    phase_took: PhaseTook
    def __init__(self, internal_response: _Optional[_Union[SearchResponseSections, _Mapping]] = ..., scroll_id: _Optional[str] = ..., point_in_time_id: _Optional[str] = ..., total_shards: _Optional[int] = ..., successful_shards: _Optional[int] = ..., skipped_shards: _Optional[int] = ..., shard_failures: _Optional[_Iterable[_Union[SearchShardFailure, _Mapping]]] = ..., clusters: _Optional[_Union[Clusters, _Mapping]] = ..., took_in_millis: _Optional[int] = ..., phase_took: _Optional[_Union[PhaseTook, _Mapping]] = ...) -> None: ...

class SearchShardFailure(_message.Message):
    __slots__ = ("reason", "target")
    REASON_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    reason: str
    target: SearchShardTarget
    def __init__(self, reason: _Optional[str] = ..., target: _Optional[_Union[SearchShardTarget, _Mapping]] = ...) -> None: ...

class SearchShardTarget(_message.Message):
    __slots__ = ("shard_id", "index_id", "node_id")
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    INDEX_ID_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    shard_id: str
    index_id: str
    node_id: str
    def __init__(self, shard_id: _Optional[str] = ..., index_id: _Optional[str] = ..., node_id: _Optional[str] = ...) -> None: ...

class Clusters(_message.Message):
    __slots__ = ("total", "successful", "skipped")
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    SUCCESSFUL_FIELD_NUMBER: _ClassVar[int]
    SKIPPED_FIELD_NUMBER: _ClassVar[int]
    total: int
    successful: int
    skipped: int
    def __init__(self, total: _Optional[int] = ..., successful: _Optional[int] = ..., skipped: _Optional[int] = ...) -> None: ...

class PhaseTook(_message.Message):
    __slots__ = ("phase_took_map",)
    class PhaseTookMapEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    PHASE_TOOK_MAP_FIELD_NUMBER: _ClassVar[int]
    phase_took_map: _containers.ScalarMap[str, int]
    def __init__(self, phase_took_map: _Optional[_Mapping[str, int]] = ...) -> None: ...

class SearchResponseSections(_message.Message):
    __slots__ = ("hits", "aggregations", "suggest", "profile_results", "timed_out", "terminated_early", "num_reduce_phases", "search_exts")
    HITS_FIELD_NUMBER: _ClassVar[int]
    AGGREGATIONS_FIELD_NUMBER: _ClassVar[int]
    SUGGEST_FIELD_NUMBER: _ClassVar[int]
    PROFILE_RESULTS_FIELD_NUMBER: _ClassVar[int]
    TIMED_OUT_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_EARLY_FIELD_NUMBER: _ClassVar[int]
    NUM_REDUCE_PHASES_FIELD_NUMBER: _ClassVar[int]
    SEARCH_EXTS_FIELD_NUMBER: _ClassVar[int]
    hits: SearchHits
    aggregations: Aggregations
    suggest: Suggest
    profile_results: SearchProfileShardResults
    timed_out: bool
    terminated_early: bool
    num_reduce_phases: int
    search_exts: _containers.RepeatedCompositeFieldContainer[SearchExtBuilder]
    def __init__(self, hits: _Optional[_Union[SearchHits, _Mapping]] = ..., aggregations: _Optional[_Union[Aggregations, _Mapping]] = ..., suggest: _Optional[_Union[Suggest, _Mapping]] = ..., profile_results: _Optional[_Union[SearchProfileShardResults, _Mapping]] = ..., timed_out: bool = ..., terminated_early: bool = ..., num_reduce_phases: _Optional[int] = ..., search_exts: _Optional[_Iterable[_Union[SearchExtBuilder, _Mapping]]] = ...) -> None: ...

class SearchHits(_message.Message):
    __slots__ = ("total_hits", "hits", "max_score", "collapse_field", "collapse_values")
    TOTAL_HITS_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    MAX_SCORE_FIELD_NUMBER: _ClassVar[int]
    COLLAPSE_FIELD_FIELD_NUMBER: _ClassVar[int]
    COLLAPSE_VALUES_FIELD_NUMBER: _ClassVar[int]
    total_hits: TotalHits
    hits: _containers.RepeatedCompositeFieldContainer[SearchHit]
    max_score: float
    collapse_field: str
    collapse_values: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, total_hits: _Optional[_Union[TotalHits, _Mapping]] = ..., hits: _Optional[_Iterable[_Union[SearchHit, _Mapping]]] = ..., max_score: _Optional[float] = ..., collapse_field: _Optional[str] = ..., collapse_values: _Optional[_Iterable[bytes]] = ...) -> None: ...

class TotalHits(_message.Message):
    __slots__ = ("value", "relation")
    class Relation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        EQUAL_TO: _ClassVar[TotalHits.Relation]
        GREATER_THAN_OR_EQUAL_TO: _ClassVar[TotalHits.Relation]
    EQUAL_TO: TotalHits.Relation
    GREATER_THAN_OR_EQUAL_TO: TotalHits.Relation
    VALUE_FIELD_NUMBER: _ClassVar[int]
    RELATION_FIELD_NUMBER: _ClassVar[int]
    value: int
    relation: TotalHits.Relation
    def __init__(self, value: _Optional[int] = ..., relation: _Optional[_Union[TotalHits.Relation, str]] = ...) -> None: ...

class SearchHit(_message.Message):
    __slots__ = ("doc_id", "score", "id", "nested_id", "version", "seq_no", "primary_term", "source", "document_fields", "meta_fields", "highlight_fields", "sort_values", "matched_queries", "shard", "index", "clusterAlias", "source_as_map")
    class DocumentFieldsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: DocumentField
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[DocumentField, _Mapping]] = ...) -> None: ...
    class MetaFieldsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: DocumentField
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[DocumentField, _Mapping]] = ...) -> None: ...
    class HighlightFieldsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: HighlightField
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[HighlightField, _Mapping]] = ...) -> None: ...
    class SourceAsMapEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bytes
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    NESTED_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SEQ_NO_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_TERM_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    META_FIELDS_FIELD_NUMBER: _ClassVar[int]
    HIGHLIGHT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    SORT_VALUES_FIELD_NUMBER: _ClassVar[int]
    MATCHED_QUERIES_FIELD_NUMBER: _ClassVar[int]
    SHARD_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    CLUSTERALIAS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_AS_MAP_FIELD_NUMBER: _ClassVar[int]
    doc_id: int
    score: float
    id: str
    nested_id: NestedIdentity
    version: int
    seq_no: int
    primary_term: int
    source: bytes
    document_fields: _containers.MessageMap[str, DocumentField]
    meta_fields: _containers.MessageMap[str, DocumentField]
    highlight_fields: _containers.MessageMap[str, HighlightField]
    sort_values: SearchSortValues
    matched_queries: _containers.RepeatedScalarFieldContainer[str]
    shard: SearchShardTarget
    index: str
    clusterAlias: str
    source_as_map: _containers.ScalarMap[str, bytes]
    def __init__(self, doc_id: _Optional[int] = ..., score: _Optional[float] = ..., id: _Optional[str] = ..., nested_id: _Optional[_Union[NestedIdentity, _Mapping]] = ..., version: _Optional[int] = ..., seq_no: _Optional[int] = ..., primary_term: _Optional[int] = ..., source: _Optional[bytes] = ..., document_fields: _Optional[_Mapping[str, DocumentField]] = ..., meta_fields: _Optional[_Mapping[str, DocumentField]] = ..., highlight_fields: _Optional[_Mapping[str, HighlightField]] = ..., sort_values: _Optional[_Union[SearchSortValues, _Mapping]] = ..., matched_queries: _Optional[_Iterable[str]] = ..., shard: _Optional[_Union[SearchShardTarget, _Mapping]] = ..., index: _Optional[str] = ..., clusterAlias: _Optional[str] = ..., source_as_map: _Optional[_Mapping[str, bytes]] = ...) -> None: ...

class NestedIdentity(_message.Message):
    __slots__ = ("field", "offset", "child")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    CHILD_FIELD_NUMBER: _ClassVar[int]
    field: str
    offset: int
    child: NestedIdentity
    def __init__(self, field: _Optional[str] = ..., offset: _Optional[int] = ..., child: _Optional[_Union[NestedIdentity, _Mapping]] = ...) -> None: ...

class DocumentField(_message.Message):
    __slots__ = ("name", "values")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    name: str
    values: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, name: _Optional[str] = ..., values: _Optional[_Iterable[bytes]] = ...) -> None: ...

class HighlightField(_message.Message):
    __slots__ = ("name", "fragments")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FRAGMENTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    fragments: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., fragments: _Optional[_Iterable[str]] = ...) -> None: ...

class SearchSortValues(_message.Message):
    __slots__ = ("formatted_sort_values", "raw_sort_values")
    FORMATTED_SORT_VALUES_FIELD_NUMBER: _ClassVar[int]
    RAW_SORT_VALUES_FIELD_NUMBER: _ClassVar[int]
    formatted_sort_values: _containers.RepeatedScalarFieldContainer[bytes]
    raw_sort_values: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, formatted_sort_values: _Optional[_Iterable[bytes]] = ..., raw_sort_values: _Optional[_Iterable[bytes]] = ...) -> None: ...

class Aggregations(_message.Message):
    __slots__ = ("aggregations",)
    AGGREGATIONS_FIELD_NUMBER: _ClassVar[int]
    aggregations: bytes
    def __init__(self, aggregations: _Optional[bytes] = ...) -> None: ...

class Suggest(_message.Message):
    __slots__ = ("suggestions",)
    SUGGESTIONS_FIELD_NUMBER: _ClassVar[int]
    suggestions: bytes
    def __init__(self, suggestions: _Optional[bytes] = ...) -> None: ...

class SearchProfileShardResults(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: bytes
    def __init__(self, results: _Optional[bytes] = ...) -> None: ...

class SearchExtBuilder(_message.Message):
    __slots__ = ("search_exts",)
    SEARCH_EXTS_FIELD_NUMBER: _ClassVar[int]
    search_exts: bytes
    def __init__(self, search_exts: _Optional[bytes] = ...) -> None: ...
