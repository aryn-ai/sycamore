import sys
import json
from sycamore.functions.simhash import shinglesDist
from remote_processors.processors.processor import ResponseProcessor
from remote_processors import SearchRequest, SearchResponse


class DedupResponseProcessor(ResponseProcessor):
    """
    DedupResponseProcessor removes lower-ranking near-duplicate documents
    from SearchResponse objects.  It requires that the `shingles` attribute
    be present in the `source` for each document.  The `threshold` parameter
    indicates how loose the definition of a near-duplicate will be.  A value
    of 0.0 requires almost exact equality.  A value of 1.0 will consider all
    documents to be the same.  Useful values seem to range from 0.1 to 0.4.
    The `verbose` parameter, if nonzero, will cause information to be printed
    about matching pairs and which documents were kept or dropped.
    """

    def __init__(self, threshold: float, verbose: int) -> None:
        super().__init__()
        self.threshold = threshold
        self.verbose = verbose

    @staticmethod
    def from_config(config) -> ResponseProcessor:
        t = config["threshold"]
        v = config.get("verbose", 0)
        return DedupResponseProcessor(threshold=t, verbose=v)

    @staticmethod
    def get_class_name() -> str:
        return "dedup-response"

    def process_response(self, req: SearchRequest, resp: SearchResponse) -> SearchResponse:
        """
        process_response does the actual work of eliminating near-duplicate
        documents.  It's basically an N-squared pairwise comparison of
        shingles.  It calls the shingles distance function from Sycamore.
        After marking documents valid or invalid, the hits array is
        rewritten with just the surviving documents.  If for any reason
        shingles can't be accessed for a document, that document will be
        marked valid.  Thus, if the query doesn't return shingles in the
        response, no documents will be dropped.
        """

        try:
            hitAry = resp.internal_response.hits.hits
            n = len(hitAry)
        except AttributeError:
            n = 0
        if n == 0:
            return resp

        validAry = [True] * n
        sketchAry = []
        for i, hit in enumerate(hitAry):
            try:
                raw_i = hit.source
                obj_i = json.loads(raw_i)
                # !!! must ask for shingles and text_representation in query
                sketch_i = obj_i.get("shingles")
                text_i = obj_i.get("text_representation", "")
            except (AttributeError, KeyError):
                sketch_i = None
                text_i = ""
            sketchAry.append(sketch_i)
            if sketch_i is None:
                continue
            for j in range(i):
                if not validAry[j]:
                    continue
                sketch_j = sketchAry[j]
                if sketch_j is None:
                    continue
                dist = shinglesDist(sketch_i, sketch_j)
                if dist < self.threshold:
                    validAry[i] = False
                    if self.verbose:
                        try:
                            raw_j = hitAry[j].source
                            obj_j = json.loads(raw_j)
                            text_j = obj_j.get("text_representation", "")
                            print("DIST", dist, file=sys.stderr)
                            print("PREV", text_j, file=sys.stderr)
                            print("CURR", text_i, file=sys.stderr)
                        except (AttributeError, KeyError):
                            pass
                    break

        del sketchAry
        newAry = []
        if not self.verbose:
            for valid, hit in zip(validAry, hitAry):
                if valid:
                    newAry.append(hit)
        else:
            for valid, hit in zip(validAry, hitAry):
                try:
                    raw = hit.source
                    obj = json.loads(raw)
                    props = obj["properties"]
                    fn = props.get("_location")
                    if fn is None:
                        fn = props["path"]
                    pn = props.get("page_number", 0)
                    name = fn + " " + str(pn)
                except (AttributeError, KeyError):
                    name = "None 0"
                if valid:
                    newAry.append(hit)
                    print("KEEP", name, file=sys.stderr)
                else:
                    print("DROP", name, file=sys.stderr)
        resp.internal_response.hits.total_hits.value = len(newAry)
        del resp.internal_response.hits.hits[:]
        resp.internal_response.hits.hits.extend(newAry)
        return resp
