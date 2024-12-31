from PIL import Image
from sycamore.data.bbox import BoundingBox
from sycamore.data.element import TableElement
from sycamore.transforms.table_structure.extract import (
    TableTransformerStructureExtractor,
    HybridTableStructureExtractor,
    DeformableTableStructureExtractor,
)


class TestTableExtractors:

    @staticmethod
    def mock_doc_image(mocker, width, height):
        im = mocker.Mock(spec=Image.Image)
        im.size = width, height
        return im

    @staticmethod
    def mock_table_element(mocker, width, height):
        elt = mocker.Mock(spec=TableElement)
        elt.bbox = BoundingBox(0, 0, width, height)
        return elt

    def test_hybrid_routing_both_gt500(self, mocker):
        im = TestTableExtractors.mock_doc_image(mocker, 1000, 1000)
        elt = TestTableExtractors.mock_table_element(mocker, 0.7, 0.7)
        extractor = HybridTableStructureExtractor(deformable_model="dont initialize me")
        chosen = extractor._pick_model(elt, im)
        assert type(chosen) == DeformableTableStructureExtractor

    def test_hybrid_routing_one_gt500(self, mocker):
        im = TestTableExtractors.mock_doc_image(mocker, 1000, 1000)
        elt = TestTableExtractors.mock_table_element(mocker, 0.7, 0.2)
        extractor = HybridTableStructureExtractor(deformable_model="dont initialize me")
        chosen = extractor._pick_model(elt, im)
        assert type(chosen) == DeformableTableStructureExtractor

    def test_hybrid_routing_neither_gt500(self, mocker):
        im = TestTableExtractors.mock_doc_image(mocker, 1000, 1000)
        elt = TestTableExtractors.mock_table_element(mocker, 0.2, 0.2)
        extractor = HybridTableStructureExtractor(deformable_model="dont initialize me")
        chosen = extractor._pick_model(elt, im)
        assert type(chosen) == TableTransformerStructureExtractor
