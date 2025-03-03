import pytest
from PIL import Image
from sycamore.data.bbox import BoundingBox
from sycamore.data.element import TableElement
from sycamore.transforms.table_structure.extract import (
    TableTransformerStructureExtractor,
    HybridTableStructureExtractor,
    DeformableTableStructureExtractor,
)

HTSE = HybridTableStructureExtractor


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
        elt.tokens = [{"text": "alrngea;rjgnekl"}]
        return elt

    def test_hybrid_routing_both_gt500(self, mocker):
        im = TestTableExtractors.mock_doc_image(mocker, 1000, 1000)
        elt = TestTableExtractors.mock_table_element(mocker, 0.7, 0.7)
        extractor = HybridTableStructureExtractor(deformable_model="dont initialize me")
        chosen = extractor._pick_model(elt, im, model_selection="pixels>500->deformable_detr;table_transformer")
        assert type(chosen) == DeformableTableStructureExtractor

    def test_hybrid_routing_one_gt500(self, mocker):
        im = TestTableExtractors.mock_doc_image(mocker, 1000, 1000)
        elt = TestTableExtractors.mock_table_element(mocker, 0.7, 0.2)
        extractor = HybridTableStructureExtractor(deformable_model="dont initialize me")
        chosen = extractor._pick_model(elt, im, model_selection="pixels>500->deformable_detr;table_transformer")
        assert type(chosen) == DeformableTableStructureExtractor

    def test_hybrid_routing_neither_gt500(self, mocker):
        im = TestTableExtractors.mock_doc_image(mocker, 1000, 1000)
        elt = TestTableExtractors.mock_table_element(mocker, 0.2, 0.2)
        extractor = HybridTableStructureExtractor(deformable_model="dont initialize me")
        chosen = extractor._pick_model(elt, im, model_selection="pixels>500->deformable_detr;table_transformer")
        assert type(chosen) == TableTransformerStructureExtractor


class TestHybridSelectionStatements:
    params = [(1000, 25), (25, 1000), (25, 25), (1000, 1000)]

    def test_static(self):
        f = HybridTableStructureExtractor.parse_model_selection("table_transformer")
        for p in self.params:
            assert f(*p) == "table_transformer"

        f = HybridTableStructureExtractor.parse_model_selection("deformable_detr ")
        for p in self.params:
            assert f(*p) == "deformable_detr"

        f = HybridTableStructureExtractor.parse_model_selection("deformable_detr; this is a comment")
        for p in self.params:
            assert f(*p) == "deformable_detr"

    def test_pixelmetric(self):
        f = HybridTableStructureExtractor.parse_model_selection("pixels > 500 -> deformable_detr; table_transformer")
        selections = [f(*p) for p in self.params]
        assert selections == ["deformable_detr", "table_transformer", "table_transformer", "deformable_detr"]

    def test_charmetric(self):
        f = HybridTableStructureExtractor.parse_model_selection("chars > 500 -> deformable_detr; table_transformer")
        selections = [f(*p) for p in self.params]
        assert selections == ["table_transformer", "deformable_detr", "table_transformer", "deformable_detr"]

    def test_bad_modelname(self):
        with pytest.raises(ValueError, match=r"Invalid statement.* model_name was not in.*"):
            HTSE.parse_model_selection("tatr")

        with pytest.raises(ValueError, match=r"Invalid statement.* Result model .* was not in.*"):
            HTSE.parse_model_selection("pixels>500 -> nonmodel")

        with pytest.raises(ValueError, match=r"Invalid statement.* model_name was not in.*"):
            HTSE.parse_model_selection("pixels>500 -> deformable_detr; yo_mama")

    def test_multiple_arrows(self):
        with pytest.raises(ValueError, match=r"Invalid statement.* Found more than 2 instances of '->'"):
            HTSE.parse_model_selection("pixels>500 -> vrooooom -> vrooooooooooom")

    def test_no_comparison(self):
        with pytest.raises(ValueError, match=r"Invalid statement.* Did not find a comparison operator .*"):
            HTSE.parse_model_selection("pixels=3->deformable_detr")

        with pytest.raises(ValueError, match=r"Invalid statement.* Did not find a comparison operator .*"):
            HTSE.parse_model_selection("chars->deformable_detr")

    def test_multiple_comparisons(self):
        with pytest.raises(ValueError, match=r"Invalid comparison.* Comparison statements must take the form .*"):
            HTSE.parse_model_selection("1000 > pixels > 300 -> deformable_detr")

    def test_backwards_comparison(self):
        with pytest.raises(ValueError, match=r"Invalid comparison.* Allowed metrics are.*"):
            HTSE.parse_model_selection("1000 > pixels -> table_transformer")

    def test_bad_metric(self):
        with pytest.raises(ValueError, match=r"Invalid comparison.* Allowed metrics are.*"):
            HTSE.parse_model_selection("pickles > 1000 -> table_transformer")

        with pytest.raises(ValueError, match=r"Invalid comparison.* Allowed metrics are.*"):
            HTSE.parse_model_selection("charm < 5 -> deformable_detr")

    def test_bad_threshold(self):
        with pytest.raises(ValueError, match=r"Invalid comparison.* Threshold .* must be numeric"):
            HTSE.parse_model_selection("pixels > chars -> table_transformer")

    def test_complicated(self):
        f = HTSE.parse_model_selection(
            "pixels>5->table_transformer; chars<30->deformable_detr;chars>35->table_transformer;"
            "pixels>2->deformable_detr;table_transformer;comment"
        )
        assert f(10, 14) == "table_transformer"
        assert f(5, 15) == "deformable_detr"
        assert f(5, 42) == "table_transformer"
        assert f(5, 32) == "deformable_detr"
        assert f(0, 32) == "table_transformer"

    def test_excess_semicolons_ok(self):
        f = HTSE.parse_model_selection("chars>0->table_transformer;")
        assert f(10, 10) == "table_transformer"

        f = HTSE.parse_model_selection(";;;chars>0->table_transformer;")
        assert f(10, 10) == "table_transformer"
