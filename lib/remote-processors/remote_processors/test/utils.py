from remote_processors.search_request_pb2 import SearchRequest
from remote_processors.search_response_pb2 import (
    SearchHit,
    SearchHits,
    SearchResponse,
    SearchResponseSections,
    SearchShardTarget,
    TotalHits,
)
from pathlib import Path


def dummy_search_request():
    return SearchRequest(
        indices=["idx1", "idx2"], search_type=SearchRequest.SearchType.QUERY_THEN_FETCH, pipeline="pipeline"
    )


def dummy_search_response():
    return SearchResponse(
        internal_response=SearchResponseSections(
            hits=SearchHits(
                total_hits=TotalHits(value=2, relation=TotalHits.Relation.EQUAL_TO),
                hits=[
                    SearchHit(
                        doc_id=0,
                        score=1,
                        id="ab005390-c05b-11ee-9502-3a049e7f7082",
                        version=-1,
                        source=b'{"doc_id":"aaffbdc2-c05b-11ee-9502-3a049e7f7082","type":"NarrativeText",\
"text_representation":"The following hardware trends allowed UNIX systems to eclipse IBM mainframes in processing \
power and sort performance. Recently, these same trends, along with \
some Windows-specific ones, have allowed the performance of large Windows servers to skyrocket.",\
"elements":[],"parent_id":\
"99140d34-c05b-11ee-9502-3a049e7f7082","properties":{"filename":"","filetype":"application/pdf","parent_id":\
"8559fa65974d3c8291e37ebc7ca2ce56","page_number":2,"links":[],"element_id":"c10210cca3b107f86c8de4abc1ed0b5d"}}',
                        shard=SearchShardTarget(
                            shard_id="[sort-benchmark][0]", index_id="sort-benchmark", node_id="bRNAxMU6TFa3dgWC382CdA"
                        ),
                    ),
                    SearchHit(
                        doc_id=1,
                        score=1,
                        id="aaffbdc2-c05b-11ee-9502-3a049e7f7082",
                        version=-1,
                        source=b'{"doc_id":"ab005390-c05b-11ee-9502-3a049e7f7082","type":"Title",\
"text_representation":"C P U b u s y %","elements":[],\
"parent_id":"99877828-c05b-11ee-9502-3a049e7f7082","properties":{"filename":"","filetype":"application/pdf",\
page_number":5,"links":[],"element_id":"92fbc8d751f7018497cba340d1676df4"}}',
                        shard=SearchShardTarget(
                            shard_id="[sort-benchmark][0]", index_id="sort-benchmark", node_id="bRNAxMU6TFa3dgWC382CdA"
                        ),
                    ),
                ],
                max_score=1,
            ),
            num_reduce_phases=1,
        ),
        total_shards=2,
        successful_shards=2,
        took_in_millis=12,
    )


TESTS_DIR = Path(__file__).parent
