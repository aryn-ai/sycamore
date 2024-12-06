from sycamore.transforms.table_structure.table_transformers import outputs_to_objects, objects_to_table
import torch
import numpy as np


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
