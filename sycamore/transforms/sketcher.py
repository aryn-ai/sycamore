import sys

from ray.data import Dataset

from sycamore.data import Document, Element
from sycamore.functions.rabin_karp import simHashText, simHashesDist
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function


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
              utf = txt.encode("utf-8")
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
        drops = 0
        ds = self.child().execute()
        ds = ds.materialize()
        ds = ds.add_column("_del", lambda _: False)
        outerIdx = 0
        for outer in ds.iter_rows():
            doc = Document(outer["doc"])
            outerSims = doc.simHashes
            if outerSims:
                innerIdx = 0
                for inner in ds.iter_rows():
                    if (innerIdx < outerIdx) and not inner["_del"]:
                        doc = Document(inner["doc"])
                        innerSims = doc.simHashes
                        if innerSims:
                            if simHashesDist(outerSims, innerSims) < 16:
                                inner["_del"] = True
                                drops += 1
                                print("Drops current", drops, file=sys.stderr)
                    innerIdx += 1
            outerIdx += 1
        ds = ds.filter(lambda row: not row["_del"])
        ds = ds.drop_columns(["_del"])
        print('Drops final', drops, file=sys.stderr)
        return ds
