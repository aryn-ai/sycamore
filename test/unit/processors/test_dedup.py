import json

from gen.search_response_pb2 import SearchHit, SearchHits, SearchResponse, SearchResponseSections, SearchShardTarget, TotalHits

from lib.processors.dedup_processor import DedupResponseProcessor


def mkShingles(x: int):
    ary = []
    for i in range(16):
        ary.append((i << 16) + (x & 1))  # shingle must be monotonic increasing
        x //= 2
    return ary


def mkJson(x: int):
    id = "%08x-%04x-%04x-%04x-%012x" % (x, x, x, x, x)
    txt = "The quick brown fox jumps over %d lazy dogs." % x
    d = {
        "doc_id": id,
        "parent_id": id,
        "type": "NarrativeText",
        "text_representation": txt,
        "elements": [],
        "properties": {
            "path": "%08x.pdf" % x,
            "page_number": x,
        },
        "shingles": mkShingles(x),
    }
    s = json.dumps(d, separators=(",", ":"))
    return s.encode("utf-8")


def mkHit(x: int):
    return SearchHit(
        doc_id=x,
        score=100 - (x / 100),
        id="%08x-%04x-%04x-%04x-%012x" % (x, x, x, x, x),
        version=-1,
        source=mkJson(x),
        shard=SearchShardTarget(
            shard_id="%x" % x,
            index_id="%x" % x,
            node_id="%022x" % x,
        ),
    )


def mkHitAry():
    return [
        mkHit(0),  # 0000
        mkHit(1),  # 0001
        mkHit(2),  # 0010
        mkHit(3),  # 0011
    ]


def mkHits():
    ary = mkHitAry()
    return SearchHits(
        total_hits=TotalHits(value=len(ary), relation=TotalHits.Relation.EQUAL_TO),
        hits=ary,
        max_score=1,
    )


def mkSearchResp():
    return SearchResponse(
        internal_response=SearchResponseSections(
            hits=mkHits(),
            num_reduce_phases=1,
        ),
        total_shards=2,
        successful_shards=2,
        took_in_millis=12,
    )


class TestDedupProcessor:

    def test_smoke(self):
        req = None
        resp = mkSearchResp()
        assert resp.internal_response.hits.total_hits.value == 4
        assert len(resp.internal_response.hits.hits) == 4
        cfg = {"threshold": 0.1}
        proc = DedupResponseProcessor.from_config(cfg)
        resp = proc.process_response(req, resp)
        hits = resp.internal_response.hits
        assert hits.total_hits.value == 2
        assert len(hits.hits) == 2
        assert hits.hits[0].doc_id == 0
        assert hits.hits[1].doc_id == 3
