from sycamore.transforms.table_structure.table_transformers import outputs_to_objects
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
