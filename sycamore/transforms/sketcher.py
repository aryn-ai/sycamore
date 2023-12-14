import sys
import re
import unicodedata

from ray.data import Dataset

from sycamore.data import Document, Element
from sycamore.functions.rabin_karp import simHashText, simHashesDist
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function


whiteRe = re.compile(r"\s+")
charMap = {
    "`": "'",
    "–": "-",
    "—": "-",
    "´": "'",
    "‘": "'",
    "’": "'",
    '“': '"',
    '”': '"',
    "Æ": "AE",
    "æ": "ae",
    "ǃ": "!",
    "™": "TM",
    "©": "(c)",
    "®": "(R)",
}


def normalizeString(s: str) -> bytes:
    s = whiteRe.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    t = ""
    for ch in s:
        t += charMap.get(ch, ch)
    return t


class Sketcher(SingleThreadUser, NonGPUUser, Transform):
    """
    For each Document, uses shingling to hash sliding windows of the
    text_representation using various permutations.  Uses SimHash
    to reduce each shingle to a similarity hash.  The set of SimHashes
    is called the sketch.  Documents' sketches can be compared to
    determine if they have near-duplicate content.

    Args:
        child: The source node or component that provides the documents

    Example:
        .. code-block:: python

            node = ...  # source node or component that provides hierarchical documents.
            xform = Sketcher(child=node)
            dataset = xform.execute()
    """

    def __init__(self, child: Node, **kwargs):
        super().__init__(child, **kwargs)

    class Callable:
        def run(self, doc: Document) -> Document:
            txt = doc.text_representation
            if txt:
              utf = normalizeString(txt).encode("utf-8")
              doc.simHashes = simHashText(utf)
            return doc


    def execute(self) -> Dataset:
        dataset = self.child().execute()
        xform = Sketcher.Callable()
        return dataset.map(generate_map_function(xform.run))

###############################################################################

class SketchUniquify(SingleThreadUser, NonGPUUser, Transform):

    def __init__(self, child: Node, **kwargs):
        super().__init__(child, **kwargs)

    def execute(self) -> Dataset:
        threshold = 13.5
        total = 0
        drops = 0
        ds = self.child().execute()
        ds = ds.materialize()
        ds = ds.add_column("_del", lambda _: False)
        seenSketches = []  # gonna use a chunk of memory
        seenText = []  # FIXME: remove
        for row in ds.iter_rows():
            total += 1
            doc = Document(row["doc"])
            docSims = doc.simHashes
            docText = normalizeString(doc.text_representation or "")
            if docSims:
                ii = 0
                for prevSims in seenSketches:
                    if prevSims:
                        dist = simHashesDist(docSims, prevSims)
                        if dist <= threshold:
                            row["_del"] = True
                            drops += 1
                            docSims = None  # don't remember deleted
                            print("[Drop]", dist, drops, total, file=sys.stderr)
                            print("[Prev]", seenText[ii], file=sys.stderr)
                            print("[Curr]", docText, file=sys.stderr)
                            break
                    ii += 1
            seenSketches.append(docSims)
            seenText.append(docText)
        seenSketches = None
        ds = ds.filter(lambda row: not row["_del"])
        ds = ds.drop_columns(["_del"])
        print('Dropped', drops, total, file=sys.stderr)
        return ds
