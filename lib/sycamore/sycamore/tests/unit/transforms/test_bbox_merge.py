import ray.data

from sycamore.data import Document
from sycamore.transforms import (
    MarkBreakByColumn,
    MarkBreakByTokens,
    MarkBreakPage,
    MarkDropHeaderFooter,
    MarkDropTiny,
    SortByPageBbox,
)
from sycamore.transforms.merge_elements import Merge, MarkedMerger
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.plan_nodes import Node


def spew(n: int, pfx: str = "") -> str:
    words = ("lorem", "ipsum", "dolor", "sit")
    rv = pfx
    for i in range(n):
        if rv:
            rv += " "
        rv += words[i % len(words)]
    return rv


def mkText(text: str, page: int, left: float, top: float, right: float, bot: float) -> dict:
    return {
        "type": "UncategorizedText",
        "text_representation": text,
        "properties": {"page_number": page, "title": "foo title"},
        "bbox": [left, top, right, bot],
    }


class FakeNode(Node):
    def __init__(self, doc: dict):
        self.doc = doc

    def execute(self, **kwargs) -> ray.data.Dataset:
        return ray.data.from_items([self.doc])


class TestBboxMerge:
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "foobar",
            "binary_representation": None,
            "parent_id": None,
            "properties": {"path": "/docs/foo.txt", "title": "bar"},
            "elements": [
                mkText("previous page", 1, 0.1, 0.8, 0.3, 0.9),
                mkText("top of page", 2, 0.1, 0.1, 0.3, 0.15),
                mkText(spew(20, "column begin"), 2, 0.1, 0.2, 0.4, 0.25),
                mkText(spew(20), 2, 0.6, 0.2, 0.9, 0.25),
                mkText(spew(20), 2, 0.1, 0.3, 0.4, 0.35),
                mkText(spew(20), 2, 0.6, 0.3, 0.9, 0.35),
                mkText(spew(20, "column end"), 2, 0.1, 0.4, 0.4, 0.45),
                mkText(spew(100, "wide text"), 2, 0.1, 0.5, 0.9, 0.55),
                mkText("0", 2, 0.5, 0.5, 0.55, 0.55),
                mkText(spew(600), 2, 0.1, 0.6, 0.9, 0.7),
                mkText("footer", 2, 0.4, 0.96, 0.6, 0.99),
                mkText("next page", 3, 0.1, 0.1, 0.3, 0.15),
            ],
        }
    )
    tokenizer = HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2")

    def testMergeElements(self):
        doc = Document(self.doc)
        doc = SortByPageBbox(None).run(doc)
        doc = MarkDropTiny(None, 2).run(doc)
        doc = MarkDropHeaderFooter(None, 0.05).run(doc)
        doc = MarkBreakPage(None).run(doc)
        doc = MarkBreakByColumn(None).run(doc)
        doc = MarkBreakByTokens(None, self.tokenizer, 512).run(doc)
        doc = MarkedMerger().merge_elements(doc)
        merged = doc.elements
        assert len(merged) == 5

        assert merged[0].text_representation.startswith("previous page")
        assert merged[0].properties["page_number"] == 1
        assert merged[0].properties["page_numbers"] == [1]
        assert merged[1].text_representation.startswith("top of page")
        assert merged[2].text_representation.startswith("wide text")
        assert merged[4].text_representation.startswith("next page")

        for elem in merged:
            assert "0" not in elem.text_representation
            assert "footer" not in elem.text_representation

    def testMergeElementsAcrossPages(self):
        doc = Document(self.doc)
        doc = SortByPageBbox(None).run(doc)
        doc = MarkDropTiny(None, 2).run(doc)
        doc = MarkDropHeaderFooter(None, 0.05).run(doc)
        doc = MarkBreakByColumn(None).run(doc)
        doc = MarkBreakByTokens(None, self.tokenizer, 512).run(doc)
        doc = MarkedMerger().merge_elements(doc)
        merged = doc.elements
        assert len(merged) == 3

        assert merged[0].text_representation.startswith("previous page")
        assert merged[0].properties["page_number"] == 1
        assert merged[0].properties["page_numbers"] == [1, 2]

        assert merged[1].text_representation.startswith("wide text")
        assert merged[1].properties["page_number"] == 2
        assert merged[1].properties["page_numbers"] == [2]

        assert merged[2].text_representation.startswith("lorem")
        assert merged[2].properties["page_number"] == 2
        assert merged[2].properties["page_numbers"] == [2, 3]

        for elem in merged:
            assert "0" not in elem.text_representation
            assert "footer" not in elem.text_representation

    def testViaExecute(self, mocker):
        node = mocker.Mock(spec=Node)
        input = ray.data.from_items([{"doc": self.doc.serialize()}])
        execMock = mocker.patch.object(node, "execute")
        execMock.return_value = input

        plan = SortByPageBbox(node)
        plan = MarkDropTiny(plan, 2)
        plan = MarkDropHeaderFooter(plan, 0.05)
        plan = MarkBreakPage(plan)
        plan = MarkBreakByColumn(plan)
        plan = MarkBreakByTokens(plan, self.tokenizer, 512)
        merger = MarkedMerger()
        merge = Merge(node, merger)
        output = merge.execute()
        output.show()
