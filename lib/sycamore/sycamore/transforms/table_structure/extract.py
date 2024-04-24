from abc import abstractmethod
from typing import Any

from PIL import Image
import pdf2image
from ray.data import Dataset
from transformers import TableTransformerForObjectDetection
from torchvision import transforms
import torch

from sycamore.data import BoundingBox, Element, Document, TableElement
from sycamore.plan_nodes import Node, Transform
from sycamore.transforms.table_structure import table_transformers
from sycamore.transforms.table_structure.table_transformers import MaxResize
from sycamore.utils.generate_ray_func import generate_map_function


class TableStructureExtractor:
    @abstractmethod
    def extract(self, element: TableElement, doc_image: Image) -> TableElement:
        pass

    def extract_from_doc(self, doc: Document) -> Document:
        """Method that extracts the table structure for each table in the Document.

        Suitable for use in a map function. The binary_representation of the doc
        should be the pdf bytes.

        This method is best effort. If a table element is missing required metadata
        it will be skipped and no error will be thrown.
        """

        # TODO: Perhaps we should support image formats in addition to PDFs.
        if doc.binary_representation is None:
            return doc

        images = pdf2image.convert_from_bytes(doc.binary_representation)
        new_elements: list[Element] = []

        for elem in doc.elements:
            if isinstance(elem, TableElement):
                if "page_number" in elem.properties:
                    page_num = elem.properties["page_number"] - 1
                elif len(images) == 1:
                    page_num = 0
                else:
                    new_elements.append(elem)
                    continue
                new_elements.append(self.extract(elem, images[page_num]))
            else:
                new_elements.append(elem)
        doc.elements = new_elements
        return doc


class TableTransformerStructureExtractor(TableStructureExtractor):
    DEFAULT_TTAR_MODEL = "microsoft/table-structure-recognition-v1.1-all"

    def __init__(self, model: str = DEFAULT_TTAR_MODEL):
        self.model = model

    # Convert tokens (text) into the format expected by the TableTransformer
    # postprocessing code.
    def _prepare_tokens(self, tokens: list[dict[str, Any]], crop_box, width, height) -> list[dict[str, Any]]:
        for i, t in enumerate(tokens):
            assert isinstance(t["bbox"], BoundingBox)
            t["bbox"].to_absolute_self(width, height).translate_self(-1 * crop_box[0], -1 * crop_box[1])
            t["bbox"] = t["bbox"].to_list()
            t["span_num"] = i
            t["line_num"] = 0
            t["block_num"] = 0
        return tokens

    def extract(self, element: TableElement, doc_image: Image.Image) -> TableElement:
        # We need a bounding box to be able to do anything.
        if element.bbox is None:
            return element

        width, height = doc_image.size

        structure_model = TableTransformerForObjectDetection.from_pretrained(self.model)

        # Crop the image to encompass just the table + some padding.
        padding = 10
        crop_box = (
            element.bbox.x1 * width - padding,
            element.bbox.y1 * height - padding,
            element.bbox.x2 * width + padding,
            element.bbox.y2 * height + padding,
        )

        cropped_image = doc_image.crop(crop_box).convert("RGB")

        # Shift the token bounding boxes to be relative to the cropped image.
        if element.tokens is not None:
            tokens = self._prepare_tokens(element.tokens, crop_box, width, height)
        else:
            tokens = []

        # Prepare the image. These magic numbers are from the TableTransformer repository.
        structure_transform = transforms.Compose(
            [MaxResize(1000), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        # Run inference using the model and convert the output to raw "objects" containing bounding boxes and types.
        pixel_values = structure_transform(cropped_image).unsqueeze(0)

        with torch.no_grad():
            outputs = structure_model(pixel_values)

        structure_id2label = structure_model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"

        objects = table_transformers.outputs_to_objects(outputs, cropped_image.size, structure_id2label)

        # Convert the raw objects to our internal table representation. This involves multiple
        # phases of postprocessing.
        table = table_transformers.objects_to_table(objects, tokens)

        # Convert cell bounding boxes to be relative to the original image.
        for cell in table.cells:
            if cell.bbox is None:
                continue

            cell.bbox.translate_self(crop_box[0], crop_box[1]).to_relative_self(width, height)

        element.table = table
        return element


DEFAULT_TABLE_STRUCTURE_EXTRACTOR = TableTransformerStructureExtractor()


class ExtractTableStructure(Transform):
    def __init__(self, child: Node, table_structure_extractor: TableStructureExtractor, **resource_args):
        super().__init__(child, **resource_args)
        self.table_structure_extractor = table_structure_extractor

    def execute(self) -> Dataset:
        input_dataset = self.child().execute()
        map_fn = generate_map_function(self.table_structure_extractor.extract_from_doc)
        dataset = input_dataset.map(map_fn)
        return dataset
