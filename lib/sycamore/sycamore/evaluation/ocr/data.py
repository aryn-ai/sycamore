from sycamore.data import Document
from PIL import Image
from datasets import load_dataset
from datasets.load import IterableDataset
from ray.data import from_huggingface, Dataset
from typing import Optional
import io
from sycamore.docset import DocSet
from sycamore.context import Context
from sycamore.plan_nodes import Scan
import json
import ast


class OCREvalDocument(Document):

    def __init__(self, document=None, /, **kwargs):
        super().__init__(document, **kwargs)
        if not document:
            document = []

    @property
    def gt_text(self):
        """
        Returns the Ground Truth text object
        """
        return self.data.get("gt_text")

    @gt_text.setter
    def gt_text(self, gt_text: str):
        """
        Sets the Ground Truth text object
        """
        self.data["gt_text"] = gt_text

    @property
    def pred_text(self):
        """
        Returns the Predicted text object
        """
        return self.data.get("pred_text")

    @pred_text.setter
    def pred_text(self, pred_text: str):
        """
        Sets the Predicted text object
        """
        self.data["pred_text"] = pred_text

    @property
    def metrics(self) -> dict:
        """Dictionary of evaluation metrics"""
        if "metrics" not in self.data:
            self.data["metrics"] = {}
        return self.data["metrics"]

    @property
    def image(self) -> Optional[Image.Image]:
        """Bytes of image of table"""
        if "image" in self.data:
            imbytes = self.data["image"]
            return Image.open(io.BytesIO(imbytes))
        return None

    @image.setter
    def image(self, im: Image.Image):
        """Set the image of the table"""
        buf = io.BytesIO()
        im.save(buf, format="png")
        self.data["image"] = buf.getvalue()


class OCREvalScan(Scan):
    def format(self):  # type: ignore
        return "OCREvalScan"

    def to_docset(self, context: Context) -> DocSet:
        return DocSet(context, self)


class InvoiceOCREvalScan(OCREvalScan):

    @staticmethod
    def _ray_row_to_document(row) -> dict[str, bytes]:
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        eval_doc = OCREvalDocument()
        eval_doc.image = img
        eval_doc.gt_text = " ".join(ast.literal_eval(json.loads(row["raw_data"])["ocr_words"]))
        return {"doc": eval_doc.serialize()}

    def execute(self, **kwargs) -> Dataset:
        hf_ds = load_dataset("mychen76/invoices-and-receipts_ocr_v1", split="train", streaming=True)
        assert isinstance(hf_ds, IterableDataset)
        ray_ds = from_huggingface(hf_ds)
        return ray_ds.map(InvoiceOCREvalScan._ray_row_to_document)


class HandwritingOCREvalScan(OCREvalScan):

    @staticmethod
    def _ray_row_to_document(row) -> dict[str, bytes]:
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        eval_doc = OCREvalDocument()
        eval_doc.image = img
        eval_doc.gt_text = row["text"]
        return {"doc": eval_doc.serialize()}

    def execute(self, **kwargs) -> Dataset:
        hf_ds = load_dataset("corto-ai/handwritten-text", split="train", streaming=True)
        assert isinstance(hf_ds, IterableDataset)
        ray_ds = from_huggingface(hf_ds)
        return ray_ds.map(HandwritingOCREvalScan._ray_row_to_document)


class BaseOCREvalScan(OCREvalScan):

    @staticmethod
    def _ray_row_to_document(row) -> dict[str, bytes]:
        img = Image.open(io.BytesIO(row["cropped_image"]["bytes"])).convert("RGB")
        eval_doc = OCREvalDocument()
        eval_doc.image = img
        eval_doc.gt_text = "".join(row["answer"]) if isinstance(row["answer"], list) else row["answer"]
        return {"doc": eval_doc.serialize()}

    def execute(self, **kwargs) -> Dataset:
        hf_ds = load_dataset("deoxykev/short_ocr_sentences", split="train", streaming=True)
        assert isinstance(hf_ds, IterableDataset)
        ray_ds = from_huggingface(hf_ds)
        return ray_ds.map(BaseOCREvalScan._ray_row_to_document)
