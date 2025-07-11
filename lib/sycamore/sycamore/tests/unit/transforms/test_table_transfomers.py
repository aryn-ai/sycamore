from sycamore.transforms.table_structure.table_transformers import (
    outputs_to_objects,
    objects_to_table,
    resolve_overlaps_func,
)
import torch
import numpy as np
import copy


class TestTableTransformers:

    def test_outputs_to_objects_invalid_bbox(self):
        class MockTensor:
            def __init__(self, data):
                self.data = data

            def detach(self):
                return self

            def cpu(self):
                return self.data

            def tolist(self):
                return self.data

        class MockMax:
            class Indices:
                def detach(self):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return np.array([[0]])

            class Values:
                def detach(self):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return np.array([[0.9]])

            indices = Indices()
            values = Values()

        class MockOutputs:
            class Logits:
                def softmax(self, dim):
                    return self

                def max(self, dim):
                    return MockMax()

            def __getitem__(self, item):
                return self.mapper[item]

            logits = Logits()
            mapper = {"pred_boxes": MockTensor([torch.tensor([[0.125, 0.3, -0.05, 0.2]])])}  # Invalid bbox: x2 < x1

        outputs = MockOutputs()

        id2label = {0: "valid_label", 1: "no object"}
        img_size = (1, 1)

        objects = outputs_to_objects(outputs, img_size, id2label)

        assert len(objects) == 0, "Invalid bbox should not be included in objects"

    def test_no_structure_but_cell_bbox(self):
        objects = [
            {
                "label": "table column",
                "score": 0.9655721783638,
                "bbox": [9.64848518371582, 11.819457054138184, 34.289146423339844, 299.254150390625],
            },
            {
                "label": "table row",
                "score": 0.8881772756576538,
                "bbox": [12.466134071350098, 104.6300048828125, 1037.6905517578125, 136.64060974121094],
                "column header": False,
            },
            {
                "label": "table row",
                "score": 0.7673768997192383,
                "bbox": [12.334745407104492, 43.644412994384766, 1037.976806640625, 74.4501724243164],
                "column header": False,
            },
            {
                "label": "table column",
                "score": 0.9910051822662354,
                "bbox": [75.4512939453125, 0, 1035.6358642578125, 0],
            },
            {
                "label": "table row",
                "score": 0.8790714740753174,
                "bbox": [12.15807819366455, 74.53314971923828, 1037.6246337890625, 105.62156677246094],
                "column header": False,
            },
            {
                "label": "table row",
                "score": 0.9501585960388184,
                "bbox": [12.459859848022461, 239.05477905273438, 1037.471923828125, 269.82366943359375],
                "column header": False,
            },
            {
                "label": "table row",
                "score": 0.8935613036155701,
                "bbox": [12.436494827270508, 178.2232208251953, 1037.421142578125, 208.3254852294922],
                "column header": False,
            },
            {
                "label": "table",
                "score": 0.9958392381668091,
                "bbox": [75.4512939453125, 0, 1035.6358642578125, 0],
                "row_column_bbox": [75.4512939453125, 0, 1035.6358642578125, 0],
            },
            {
                "label": "table row",
                "score": 0.8249932527542114,
                "bbox": [12.454593658447266, 148.0679931640625, 1037.531982421875, 177.94284057617188],
                "column header": False,
            },
            {
                "label": "table row",
                "score": 0.9756750464439392,
                "bbox": [12.40351390838623, 269.0418701171875, 1037.6041259765625, 299.9597473144531],
                "column header": False,
            },
            {
                "label": "table row",
                "score": 0.5053465962409973,
                "bbox": [12.367852210998535, 12.36871337890625, 1037.98388671875, 42.043609619140625],
                "column header": False,
            },
            {
                "label": "table row",
                "score": 0.942719042301178,
                "bbox": [12.402883529663086, 208.52511596679688, 1037.5247802734375, 239.34768676757812],
                "column header": False,
            },
        ]
        tokens = [
            {
                "text": "Something really long in the original",
                "bbox": [14.906158447265625, 16.469563589409574, 1038.4617140028206, 334.5223640894096],
                "span_num": 0,
                "line_num": 0,
                "block_num": 0,
            }
        ]

        table = objects_to_table(objects, tokens)
        assert table is not None
        assert all(c.bbox is not None for c in table.cells)


class TestResolveOverlapsFunc:
    def test_empty_list(self) -> None:
        objects: list[dict] = []
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=False)
        assert resolved_objects == []

    def test_single_element_list(self):
        objects = [{"bbox": [0, 0, 10, 10]}]
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=False)
        assert resolved_objects == objects

    def test_no_overlap_cols(self):
        objects = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [11, 0, 20, 10]},
            {"bbox": [21, 0, 30, 10]},
        ]
        objects_copy = copy.deepcopy(objects)
        resolved_objects = resolve_overlaps_func(objects_copy, is_row=False)
        assert resolved_objects == objects

    def test_no_overlap_rows(self):
        objects = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [0, 11, 10, 20]},
            {"bbox": [0, 21, 10, 30]},
        ]
        objects_copy = copy.deepcopy(objects)
        resolved_objects = resolve_overlaps_func(objects_copy, is_row=True)
        assert resolved_objects == objects

    def test_adjacent_overlap_cols(self):
        objects = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [9, 0, 20, 10]},
        ]
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=False)
        assert resolved_objects[0]["bbox"] == [0, 0, 9.5, 10]
        assert resolved_objects[1]["bbox"] == [9.5, 0, 20, 10]

    def test_adjacent_overlap_rows(self):
        objects = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [0, 9, 10, 20]},
        ]
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=True)
        assert resolved_objects[0]["bbox"] == [0, 0, 10, 9.5]
        assert resolved_objects[1]["bbox"] == [0, 9.5, 10, 20]

    def test_multiple_adjacent_overlaps_cols(self):
        objects = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [8, 0, 20, 10]},
            {"bbox": [18, 0, 30, 10]},
        ]
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=False)
        assert resolved_objects[0]["bbox"] == [0, 0, 9.0, 10]
        assert resolved_objects[1]["bbox"] == [9.0, 0, 19.0, 10]
        assert resolved_objects[2]["bbox"] == [19.0, 0, 30, 10]

    def test_non_adjacent_overlap_cols(self):
        # obj 0 overlaps obj 2. obj 2 start should be pushed to obj 1 end.
        objects = [
            {"bbox": [0, 0, 12, 10]},
            {"bbox": [11, 0, 20, 10]},
            {"bbox": [10, 0, 30, 10]},
        ]
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=False)
        assert resolved_objects[0]["bbox"] == [0, 0, 11.5, 10]
        assert resolved_objects[1]["bbox"] == [11.5, 0, 20, 10]
        assert resolved_objects[2]["bbox"] == [20, 0, 30, 10]

    def test_non_adjacent_overlap_rows(self):
        objects = [
            {"bbox": [0, 0, 10, 12]},
            {"bbox": [0, 11, 10, 20]},
            {"bbox": [0, 10, 10, 30]},
        ]
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=True)
        assert resolved_objects[0]["bbox"] == [0, 0, 10, 11.5]
        assert resolved_objects[1]["bbox"] == [0, 11.5, 10, 20]
        assert resolved_objects[2]["bbox"] == [0, 20, 10, 30]

    def test_collapse_object_cols(self):
        objects = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [5, 0, 7, 10]},
        ]
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=False)
        assert resolved_objects[0]["bbox"] == [0, 0, 7.5, 10]
        assert resolved_objects[1]["bbox"] == [7, 0, 7, 10]
        assert resolved_objects[1]["bbox"][0] == resolved_objects[1]["bbox"][2]

    def test_collapse_object_rows(self):
        objects = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [0, 5, 10, 7]},
        ]
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=True)
        assert resolved_objects[0]["bbox"] == [0, 0, 10, 7.5]
        assert resolved_objects[1]["bbox"] == [0, 7, 10, 7]
        assert resolved_objects[1]["bbox"][1] == resolved_objects[1]["bbox"][3]

    def test_collapse_object_non_adjacent_cols(self):
        objects = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [11, 0, 15, 10]},
            {"bbox": [9, 0, 12, 10]},
        ]
        resolved_objects = resolve_overlaps_func(copy.deepcopy(objects), is_row=False)
        assert resolved_objects[0]["bbox"] == [0, 0, 10, 10]
        assert resolved_objects[1]["bbox"] == [11, 0, 13.5, 10]
        assert resolved_objects[2]["bbox"] == [12, 0, 12, 10]
        assert resolved_objects[2]["bbox"][0] == resolved_objects[2]["bbox"][2]
