from abc import abstractmethod
from typing import Any, Union, Optional, Callable

from PIL import Image
import pdf2image

from sycamore.data import BoundingBox, Element, Document, TableElement
from sycamore.data.document import DocumentPropertyTypes
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.transforms.table_structure import table_transformers
from sycamore.transforms.table_structure.table_transformers import MaxResize
from sycamore.utils.time_trace import timetrace
from sycamore.utils import choose_device
from sycamore.utils.import_utils import requires_modules

Num = Union[float, int]


class TableStructureExtractor:
    """Interface for extracting table structure."""

    @abstractmethod
    def extract(self, element: TableElement, doc_image: Image.Image) -> TableElement:
        """Extracts the table structure from the specified element.

        Takes a TableElement containing a bounding box, for example from the SycamorePartitioner,
        and populates the table property with information about the cells.

        Args:
          element: A TableElement.
          doc_image: A PIL object containing an image of the Document page containing the element.
               Used for bounding box calculations.
        """
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
                if DocumentPropertyTypes.PAGE_NUMBER in elem.properties:
                    page_num = elem.properties[DocumentPropertyTypes.PAGE_NUMBER] - 1
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
    """A TableStructureExtractor implementation that uses the the TableTransformer model.

    More information about TableTransformers can be found at https://github.com/microsoft/table-transformer.
    """

    DEFAULT_TATR_MODEL = "microsoft/table-structure-recognition-v1.1-all"

    def __init__(self, model: str = DEFAULT_TATR_MODEL, device=None):
        """
        Creates a TableTransformerStructureExtractor

        Args:
          model: The HuggingFace URL for the TableTransformer model to use.
        """

        self.model = model
        self.device = device
        self.structure_model = None

    def _get_device(self) -> str:
        return choose_device(self.device)

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

    def _init_structure_model(self):
        from transformers import TableTransformerForObjectDetection

        self.structure_model = TableTransformerForObjectDetection.from_pretrained(self.model).to(self._get_device())

    @timetrace("tblExtr")
    @requires_modules(["torch", "torchvision"], extra="local-inference")
    def extract(
        self, element: TableElement, doc_image: Image.Image, union_tokens=False, apply_thresholds=False
    ) -> TableElement:
        """Extracts the table structure from the specified element using a TableTransformer model.

        Takes a TableElement containing a bounding box, for example from the SycamorePartitioner,
        and populates the table property with information about the cells.

        Args:
          element: A TableElement. The bounding box must be non-null.
          doc_image: A PIL object containing an image of the Document page containing the element.
               Used for bounding box calculations.
          union_tokens: Make sure that ocr/pdfminer tokens are _all_ included in the table.
          apply_thresholds: Apply class thresholds to the objects output by the model.
        """

        # We need a bounding box to be able to do anything.
        if element.bbox is None:
            return element

        from torchvision import transforms

        width, height = doc_image.size

        if self.structure_model is None:
            self._init_structure_model()
        assert self.structure_model is not None  # For typechecking

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
        pixel_values = structure_transform(cropped_image).unsqueeze(0).to(self._get_device())

        import torch

        with torch.no_grad():
            outputs = self.structure_model(pixel_values)

        structure_id2label = self.structure_model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"

        objects = table_transformers.outputs_to_objects(
            outputs, cropped_image.size, structure_id2label, apply_thresholds=apply_thresholds
        )

        # Convert the raw objects to our internal table representation. This involves multiple
        # phases of postprocessing.
        table = table_transformers.objects_to_table(objects, tokens, union_tokens=union_tokens)

        if table is None:
            element.table = None
            return element

        # Convert cell bounding boxes to be relative to the original image.
        for cell in table.cells:
            if cell.bbox is None:
                continue

            cell.bbox.translate_self(crop_box[0], crop_box[1]).to_relative_self(width, height)

        element.table = table
        return element


class DeformableTableStructureExtractor(TableTransformerStructureExtractor):
    """A TableStructureExtractor implementation that uses the Deformable DETR model."""

    def __init__(self, model: str, device=None):
        """
        Creates a TableTransformerStructureExtractor

        Args:
          model: The HuggingFace URL or local path for the DeformableDETR model to use.
        """

        super().__init__(model, device)

    def _init_structure_model(self):
        from sycamore.utils.model_load import load_deformable_detr

        self.structure_model = load_deformable_detr(self.model, self._get_device())

    def _get_device(self) -> str:
        return choose_device(self.device, detr=True)

    def extract(
        self, element: TableElement, doc_image: Image.Image, union_tokens=False, apply_thresholds=True
    ) -> TableElement:
        """Extracts the table structure from the specified element using a DeformableDETR model.

        Takes a TableElement containing a bounding box, for example from the SycamorePartitioner,
        and populates the table property with information about the cells.

        Args:
          element: A TableElement. The bounding box must be non-null.
          doc_image: A PIL object containing an image of the Document page containing the element.
               Used for bounding box calculations.
          union_tokens: Make sure that ocr/pdfminer tokens are _all_ included in the table.
          apply_thresholds: Apply class thresholds to the objects output by the model.
        """
        # Literally just call the super but change the default for apply_thresholds
        return super().extract(element, doc_image, union_tokens, apply_thresholds)


class HybridTableStructureExtractor(TableStructureExtractor):
    """A TableStructureExtractor implementation that conditionally uses either Deformable or TATR
    depending on the size of the table"""

    _model_names = ("table_transformer", "deformable_detr")
    _metrics = ("pixels", "chars")
    _comparisons = (
        "==",
        "<=",
        ">=",
        "!=",
        "<",
        ">",
    )

    def __init__(
        self,
        deformable_model: str,
        tatr_model: str = TableTransformerStructureExtractor.DEFAULT_TATR_MODEL,
        device=None,
    ):
        self._deformable = DeformableTableStructureExtractor(deformable_model, device)
        self._tatr = TableTransformerStructureExtractor(tatr_model, device)

    def _pick_model(
        self,
        element: TableElement,
        doc_image: Image.Image,
        model_selection: str,
    ) -> Union[TableTransformerStructureExtractor, DeformableTableStructureExtractor]:
        """Use the model_selection expression to choose the model to use for table extraction.
        If the expression returns None, use table transformer."""
        if element.bbox is None:
            return self._tatr

        select_fn = self.parse_model_selection(model_selection)

        width, height = doc_image.size
        bb = element.bbox.to_absolute(width, height)
        padding = 10
        max_dim = max(bb.width, bb.height) + 2 * padding

        nchars = sum(len(tok["text"]) for tok in element.tokens or [{"text": ""}])

        selection = select_fn(max_dim, nchars)
        print("=" * 80)
        print(selection)
        if selection == "table_transformer":
            return self._tatr
        elif selection == "deformable_detr":
            return self._deformable
        elif selection is None:
            return self._tatr
        raise ValueError(f"Somehow we got an invalid selection: {selection}. This should be unreachable.")

    def _init_structure_model(self):
        self._deformable._init_structure_model()
        self._tatr._init_structure_model()

    def extract(
        self,
        element: TableElement,
        doc_image: Image.Image,
        union_tokens=False,
        model_selection: str = "pixels > 500 -> deformable_detr; table_transformer",
    ) -> TableElement:
        """Extracts the table structure from the specified element using a either a DeformableDETR or
        TATR model, depending on the size of the table.

        Takes a TableElement containing a bounding box, for example from the SycamorePartitioner,
        and populates the table property with information about the cells.

        Args:
          element: A TableElement. The bounding box must be non-null.
          doc_image: A PIL object containing an image of the Document page containing the element.
               Used for bounding box calculations.
          union_tokens: Make sure that ocr/pdfminer tokens are _all_ included in the table.
          apply_thresholds: Apply class thresholds to the objects output by the model.
          model_selection: Control which model gets selected. See ``parse_model_selection`` for
                expression syntax. Default is "pixels > 500 -> deformable_detr; table_transformer".
                If no statements are matched, defaults to table transformer.
        """
        m = self._pick_model(element, doc_image, model_selection)
        return m.extract(element, doc_image, union_tokens)

    @classmethod
    def parse_model_selection(cls, selection: str) -> Callable[[float, int], Optional[str]]:
        """
        Parse a model selection expression. Model selection expressions are of the form:
            "metric cmp threshold -> model; metric cmp threshold -> model; model;"
        That is, any number of conditional expression selections followed by up to one unconditional
        selection expression, separated by semicolons. Expressions are processed from left to right.
        Anything after the unconditional expression is not processed.

        - Supported metrics are "pixels" - the number of pixels in the larger dimension of the table (we
        find this to be easier to reason about than the total number of pixels which depends on two numbers),
        and "chars" - the number of characters in the table, as detected by the partitioner's text_extractor.
        - Supported comparisons are the usual set - <, >, <=, >=, ==, !=.
        - The threshold must be numeric (and int or a float)
        - The model must be either "deformable_detr" or "table_transformer"

        Args:
            selection: the selection string.

        Returns:
            a function that can be used to select a model given the pixels and chars metrics.

        Examples:
            - `"table_transformer"` => always use table transformer
            - `"pixels > 500 -> deformable_detr; table_transformer"` => if the biggest dimension of
                the table is greater than 500 pixels use deformable detr. Otherwise use table_transformer.
            - `"pixels>50->table_transformer; chars<30->deformable_detr;chars>35->table_transformer;pixels>2->deformable_detr;table_transformer;comment"`
                => if the biggest dimension is more than 50 pixels use table transformer. Else if the total number of chars in the table is less than
                30 use deformable_detr. Else if there are mode than 35 chars use table transformer. Else if there are more than 2 pixels in the biggest
                dimension use deformable detr. Otherwise use table transformer. comment is not processed.
        """  # noqa: E501 # line too long. long line is a long example. I want it that way.
        statements = selection.split(sep=";")
        checks = []
        for statement in statements:
            statement = statement.strip()
            if statement == "":
                continue
            if "->" not in statement:
                if statement not in cls._model_names:
                    raise ValueError(
                        f"Invalid statement: '{statement}'. Did not find '->', so this is assumed"
                        f" to be a static statement, but the model_name was not in {cls._model_names}"
                    )
                checks.append(lambda pixels, chars: statement)
                break
            pieces = statement.split(sep="->")
            if len(pieces) > 2:
                raise ValueError(f"Invalid statement: '{statement}'. Found more than 2 instances of '->'")
            result = pieces[1].strip()
            if result not in cls._model_names:
                raise ValueError(
                    f"Invalid statement: '{statement}'. Result model ({result}) was not in {cls._model_names}"
                )
            if all(c not in pieces[0] for c in cls._comparisons):
                raise ValueError(
                    f"Invalid statement: '{statement}'. Did not find a comparison operator {cls._comparisons}"
                )
            metric, cmp, threshold = cls.parse_comparison(pieces[0])

            def make_check(metric, compare, threshold, result):
                # otherwise captrued variables change their values
                def check(pixels: float, chars: int) -> Optional[str]:
                    if metric == "pixels":
                        cmpval = pixels
                    else:
                        cmpval = chars
                    if compare(cmpval, threshold):
                        return result
                    return None

                return check

            checks.append(make_check(metric, cmp, threshold, result))

        def select_fn(pixels: float, chars: int) -> Optional[str]:
            for c in checks:
                if (rv := c(pixels, chars)) is not None:
                    return rv
            return None

        return select_fn

    @staticmethod
    def get_cmp_fn(opstring: str) -> Callable[[Num, Num], bool]:
        ops = {
            "!=": lambda a, b: a != b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
        }
        if opstring in ops:
            return ops[opstring]
        raise ValueError(f"Invalid comparison: Unsupported operator '{opstring}'")

    @classmethod
    def parse_comparison(cls, comparison: str) -> tuple[str, Callable[[Num, Num], bool], Union[int, float]]:
        cmp_pieces = []
        cmp = None
        for opstring in sorted(cls._comparisons, key=lambda c: len(c), reverse=True):
            if opstring in comparison:
                cmp_pieces = comparison.split(sep=opstring)
                cmp = cls.get_cmp_fn(opstring)
                break

        if len(cmp_pieces) == 0 or cmp is None:
            raise ValueError(
                f"Invalid comparison: '{comparison}'. Did not find comparison operator {cls._comparisons}."
            )
        if len(cmp_pieces) != 2:
            raise ValueError(
                f"Invalid comparison: '{comparison}'. Comparison statements must take the form 'METRIC CMP THRESHOLD'."
            )
        metric, threshold = cmp_pieces[0].strip(), cmp_pieces[1].strip()
        if metric not in cls._metrics:
            raise ValueError(f"Invalid comparison: '{comparison}'. Allowed metrics are: '{cls._metrics}'")
        try:
            threshold_num: Num = int(threshold)
        except ValueError:
            try:
                threshold_num = float(threshold)
            except ValueError:
                raise ValueError(f"Invalid comparison: '{comparison}'. Threshold ({threshold}) must be numeric")
        return metric, cmp, threshold_num


DEFAULT_TABLE_STRUCTURE_EXTRACTOR = TableTransformerStructureExtractor


class ExtractTableStructure(Map):
    """ExtractTableStructure is a transform class that extracts table structure from a document.

    Note that this transform is for extracting the structure of tables that have already been
    identified and stored as TableElements. This is typically done with a segementation
    model, such as the one used by the SycamorePartitioner.

    When using the SycamorePartitioner, you can extract the table structure as part of
    the partitioning process by passing in extract_table_structure=True. This transform
    is provided for the rare cases in which you are using a different partitioner, but
    still need to extract the table structure.

    Args:
      child: The source node producing documents with tables to extract.
      table_structure_extractor: The extractor to use to get the table structure.
      resource_args: Additional resource related arguments to pass to the underlying runtime.

    """

    def __init__(self, child: Node, table_structure_extractor: TableStructureExtractor, **resource_args):
        super().__init__(child, f=table_structure_extractor.extract_from_doc, **resource_args)
