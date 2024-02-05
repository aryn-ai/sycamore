import sys
import re
import functools
import unicodedata

from ray.data import ActorPoolStrategy, Dataset

from sycamore.data import Document
from sycamore.functions.simhash import shinglesCalc, shinglesDist
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function
from sycamore.utils.generate_ray_func import generate_map_batch_filter_class_from_callable


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
    text_representation.  The set of shingles is called the sketch.
    Documents' sketches can be compared to determine if they have
    near-duplicate content.  The SketchUniquify transform can be used
    to de-duplicate small docsets in Sycamore. De-duplicating at
    retrieval-time is more scalable and avoids some relevance problems.

    Args:
        child: The source node or component that provides the documents
        window: Number of bytes in the sliding window that is hashed (17)
        number: Count of hashes comprising a shingle (16)

    Example:
        .. code-block:: python

            node = ...  # source node or component that provides hierarchical documents.
            xform = Sketcher(child=node)
            dataset = xform.execute()
    """

    def __init__(self, child: Node, window: int = 17, number: int = 16, **kwargs):
        super().__init__(child, **kwargs)
        self.window = window
        self.number = number

    class Callable:
        def __init__(self, window: int, number: int):
            self.window = window
            self.number = number

        def run(self, doc: Document) -> Document:
            txt = doc.text_representation
            if txt:
                utf = normalizeString(txt).encode("utf-8")
                doc.shingles = shinglesCalc(utf, self.window, self.number)
            return doc

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        xform = Sketcher.Callable(self.window, self.number)
        return dataset.map(generate_map_function(xform.run))


class SketchUniquify(SingleThreadUser, NonGPUUser, Transform):
    """
    Removes each Document which is a near-duplicate of a Document seen
    before.  Uses the shingles calculated by the Sketcher transform.
    This approach requires full materialization of the entire docset on a
    single node.  It will store all sketches in memory.  It is not
    suitable for large docsets.

    Args:
        child: The source node or component that provides the documents
        threshold: Largest distance to be considered a duplicate (0.4)

    Example:
        .. code-block:: python

           node = ...  # source node
           xform = SketchUniquify(child=node)
           dataset = xform.execute()
    """

    def __init__(self, child: Node, threshold: float = 0.4, **kwargs) -> None:
        super().__init__(child, **kwargs)
        self.threshold = threshold

    class Predicate:
        def __init__(self, threshold: float) -> None:
            self.threshold = threshold
            self.total = 0
            self.drops = 0
            # This is a significant amount of memory...
            self.seenSketches: list[list[int]] = []

        def good(self, doc: Document) -> bool:
            self.total += 1
            docSketch = doc.shingles
            if docSketch:
                for prevSketch in self.seenSketches:
                    dist = shinglesDist(docSketch, prevSketch)
                    if dist <= self.threshold:
                        self.drops += 1
                        print(f"SketchUniquify dropped {self.drops} of {self.total}", file=sys.stderr)
                        return False
                self.seenSketches.append(docSketch)
            return True

    def execute(self) -> Dataset:
        ds = self.child().execute()
        ds = ds.materialize()  # force previous to finish to free up memory
        pred = SketchUniquify.Predicate(self.threshold)

        filter_class = generate_map_batch_filter_class_from_callable(pred.good)

        # Size is 1 here to use a global view of previous sketches...
        ds = ds.map_batches(filter_class, compute=ActorPoolStrategy(size=1))
        ds = ds.materialize()  # force filter to finish before moving on
        return ds


class SketchDebug(SingleThreadUser, NonGPUUser, Transform):
    """
    Removes each Document which is a near-duplicate of a Document seen
    before.  Prints out duplicate pairs and a histogram of distances.
    Uses the shingles calculated by the Sketcher transform.
    This approach requires full materialization of the entire docset on a
    single node.  It will store all sketches in memory.  It is not
    suitable for large docsets.

    Args:
        child: The source node or component that provides the documents
        threshold: Largest distance to be considered a duplicate (0.4)

    Example:
        .. code-block:: python

           node = ...  # source node
           xform = SketchUniquify(child=node)
           dataset = xform.execute()
    """

    def __init__(self, child: Node, threshold: float = 0.4, **kwargs) -> None:
        super().__init__(child, **kwargs)
        self.threshold = threshold

    class Predicate:
        def __init__(self, threshold: float) -> None:
            self.threshold = threshold
            self.total = 0
            self.drops = 0
            # This is a significant amount of memory...
            self.seenSketches: list[list[int]] = []
            self.seenText: list[str] = []
            self.seenLoc: list[str] = []
            self.hist: dict[int, int] = {}

        def good(self, doc: Document) -> bool:
            self.total += 1
            docSketch = doc.shingles
            docText = str(doc.text_representation)
            docLoc = "%s:%s" % (doc.properties.get("path", "NoPath"), doc.properties.get("page_number", "NoPage"))
            if docSketch:
                for ii, prevSketch in enumerate(self.seenSketches):
                    dist = shinglesDist(docSketch, prevSketch)
                    dpct = int(dist * 100.0)
                    self.hist[dpct] = 1 + self.hist.get(dpct, 0)
                    if dist <= self.threshold:
                        self.drops += 1
                        print(f"SketchDebug dropped {self.drops} of {self.total}", file=sys.stderr)
                        seenText = self.seenText[ii]
                        seenLoc = self.seenLoc[ii]
                        print("DIST", dist, file=sys.stderr)
                        print("PREV", seenLoc, seenText, file=sys.stderr)
                        print("CURR", docLoc, docText, file=sys.stderr)
                        print(
                            functools.reduce(
                                lambda s, k: s + "%d=%d " % (k, self.hist[k]), sorted(self.hist.keys()), ""
                            ),
                            file=sys.stderr,
                        )
                        return False
                self.seenSketches.append(docSketch)
                self.seenText.append(docText)
                self.seenLoc.append(docLoc)
            return True

    def execute(self) -> Dataset:
        ds = self.child().execute()
        ds = ds.materialize()  # force previous to finish to free up memory
        pred = SketchDebug.Predicate(self.threshold)

        filter_class = generate_map_batch_filter_class_from_callable(pred.good)

        # Size is 1 here to use a global view of previous sketches...
        ds = ds.map_batches(filter_class, compute=ActorPoolStrategy(size=1))
        ds = ds.materialize()  # force filter to finish before moving on
        return ds
