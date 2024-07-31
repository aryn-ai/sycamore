from abc import ABC, abstractmethod
import apted
import distance
from typing import Optional, Callable

from ray.data.aggregate import AggregateFn

from sycamore.data.document import Document, MetadataDocument
from sycamore.data.table import Table
from sycamore.evaluation.tables.benchmark_scans import TableEvalDoc


def apply_metric(metric: "TableComparisonMetric") -> Callable[[Document], Document]:
    def f(doc: Document):
        ed = TableEvalDoc(doc.data)
        if ed.pred_table is None or ed.gt_table is None:
            score = 0.0
        else:
            score = metric.score(ed.gt_table, ed.pred_table)
        ed.metrics[metric.get_name()] = score
        return ed

    f.__name__ = metric.get_name()
    return f


class TableComparisonMetric(ABC):

    @abstractmethod
    def score(self, gt_table: Table, pred_table: Table) -> float:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def to_aggregate_fn(self) -> AggregateFn:
        def init(k):
            return 0, 0

        def acc_row(agg, row):
            ed = Document.deserialize(row["doc"])
            if isinstance(ed, MetadataDocument):
                return agg
            ed = TableEvalDoc(ed)
            score = ed.metrics.get(self.get_name(), 0)
            return agg[0] + score, agg[1] + 1

        def merge(agg1, agg2):
            return agg1[0] + agg2[0], agg1[1] + agg2[1]

        def finalize(agg):
            return agg[0] / agg[1]

        return AggregateFn(init=init, name=self.get_name(), merge=merge, accumulate_row=acc_row, finalize=finalize)


class TEDSMetric(TableComparisonMetric):
    def __init__(self, structure_only: bool = True):
        self._structure_only = structure_only

    def score(self, gt_table: Table, pred_table: Table) -> float:
        gt_tt = TableTree.from_table(gt_table)
        pred_tt = TableTree.from_table(pred_table)

        dist = apted.APTED(gt_tt, pred_tt, config=TEDSConfig(self._structure_only)).compute_edit_distance()
        return 1.0 - float(dist) / max(gt_tt.get_size(), pred_tt.get_size(), 1)

    def get_name(self) -> str:
        if self._structure_only:
            return "TEDS-Struct"
        else:
            return "TEDS"


class TEDSConfig(apted.Config):
    def __init__(self, structure_only: bool):
        self._structure_only = structure_only

    def rename(self, node1, node2):  # type: ignore # int is expected but float is more useful, and python exists
        if node1.tag != node2.tag or node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0
        if not self._structure_only and node1.tag == "td":
            if node1.text is not None and node2.text is not None:
                dist = distance.nlevenshtein(node1.text, node2.text, method=1)
                return dist
            return 1.0
        return 0.0


class TableTree(apted.helpers.Tree):
    def __init__(
        self,
        tag: str,
        colspan: Optional[int] = None,
        rowspan: Optional[int] = None,
        text: Optional[str] = None,
        children: Optional[list["TableTree"]] = None,
    ):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.text = text
        if children is None:
            self.children = []
        else:
            self.children = children

    @staticmethod
    def from_table(table: Table) -> "TableTree":
        root = TableTree(tag="table")
        if len(table.cells) == 0:
            return root

        curr_row = 0
        row = TableTree(tag="tr")
        root.children.append(row)

        # TODO: We should eventually put these in <thead> and <tbody> tags.
        for cell in table.cells:

            rowspan = len(cell.rows)
            colspan = len(cell.cols)

            if cell.rows[0] > curr_row:
                curr_row = cell.rows[0]
                row = TableTree(tag="tr")
                root.children.append(row)

            leaf_tag = "th" if cell.is_header else "td"
            tcell = TableTree(tag=leaf_tag, rowspan=rowspan, colspan=colspan, text=cell.content)
            row.children.append(tcell)
        return root

    def bracket(self) -> str:
        """Return the bracket format of this tree, which is what apted expects."""

        if self.tag in {"td", "th"}:
            result = f'"tag": {self.tag}, "colspan": {self.colspan}, "rowspan": {self.rowspan}, "text": {self.text}'
        else:
            result = f'"tag": {self.tag}'
        result += "".join(child.bracket() for child in self.children)
        return "{{{}}}".format(result)

    def get_size(self) -> int:
        return 1 + sum(child.get_size() for child in self.children)

    def to_html(self):
        if self.text:
            assert len(self.children) == 0, f"Found text in a non leaf node??? {self.bracket()}"
            return f'<{self.tag} colspan="{self.colspan}" rowspan="{self.rowspan}">{self.text}</{self.tag}>'
        else:
            return f'<{self.tag}>{"".join(c.to_html() for c in self.children)}</{self.tag}>'
