import sys
import re
import unicodedata

from typing import Any, Optional

from ray.data import ActorPoolStrategy, Dataset

from sycamore.data import Document
from sycamore.functions.simhash import simHashText, simHashesDist
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function


unwantedRe = re.compile(r"\W+")


def normalizeString(s: str) -> str:
    """
    Removes all non-word-constituent characters, converts Unicode
    to composed form, and changes to lowercase.
    """

    s = unwantedRe.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    return s.lower()


class Sketcher(SingleThreadUser, NonGPUUser, Transform):
    """
    For each Document, uses shingling to hash sliding windows of the
    text_representation using various permutations.  Uses SimHash
    to reduce each shingle to a similarity hash.  The set of SimHashes
    is called the sketch.  Documents' sketches can be compared to
    determine if they have near-duplicate content.  The SketchUniquify
    transform can be used to de-duplicate small docsets in Sycamore.
    De-duplicating at retrieval-time is more scalable and avoids some
    relevance problems.

    Args:
        child: The source node or component that provides the documents
        window: Number of bytes in the sliding window that is hashed (32)
        courses: Number of hashes comprising a shingle (15, must be odd)
        tabs: Number of permutation variants in each shingle (8)

    Example:
        .. code-block:: python

            node = ...  # source node or component that provides hierarchical documents.
            xform = Sketcher(child=node)
            dataset = xform.execute()
    """

    def __init__(self, child: Node, window: int = 32, courses: int = 15, tabs: int = 8, **kwargs):
        super().__init__(child, **kwargs)
        self.window = window
        self.courses = courses
        self.tabs = tabs

    class Callable:
        def __init__(self, window: int, courses: int, tabs: int):
            self.window = window
            self.courses = courses
            self.tabs = tabs

        def run(self, doc: Document) -> Document:
            txt = doc.text_representation
            if txt:
                utf = normalizeString(txt).encode("utf-8")
                doc.simHashes = simHashText(utf, self.window, self.courses, self.tabs)
            return doc

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        xform = Sketcher.Callable(self.window, self.courses, self.tabs)
        return dataset.map(generate_map_function(xform.run))


class SketchUniquify(SingleThreadUser, NonGPUUser, Transform):
    """
    Removes each Document which is a near-duplicate of a Document seen
    before.  Uses the SimHash values calculated by the Sketcher transform.
    This approach requires full materialization of the entire docset on a
    single node.  It will store all sketches in memory.  It is not
    suitable for large docsets.

    Args:
        child: The source node or component that provides the documents
        threshold: Largest distance to be considered a duplicate (16)

    Example:
        .. code-block:: python

           node = ...  # source node
           xform = SketchUniquify(child=node)
           dataset = xform.execute()
    """

    def __init__(self, child: Node, threshold: float = 16, **kwargs) -> None:
        super().__init__(child, **kwargs)
        self.threshold = threshold

    class Predicate:
        def __init__(self, threshold: float) -> None:
            self.threshold = threshold
            self.total = 0
            self.drops = 0
            # This is a significant amount of memory...
            self.seenSketches: list[list[int]] = []

        def good(self, row: dict[str, Any]) -> bool:
            self.total += 1
            doc = Document.from_row(row)
            docSims = doc.simHashes
            if docSims:
                for prevSims in self.seenSketches:
                    dist = simHashesDist(docSims, prevSims)
                    if dist <= self.threshold:
                        self.drops += 1
                        print(f"SketchUniquify dropped {self.drops} of {self.total}",
                              file=sys.stderr)
                        return False
                self.seenSketches.append(docSims)
            return True

    def execute(self) -> Dataset:
        ds = self.child().execute()
        ds = ds.materialize()  # force previous to finish to free up memory
        pred = SketchUniquify.Predicate(self.threshold)
        # Size is 1 here to use a global view of previous sketches...
        ds = ds.filter(pred.good, compute=ActorPoolStrategy(size=1))
        ds = ds.materialize()  # force filter to finish before moving on
        return ds
