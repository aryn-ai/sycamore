from typing import Iterable, Optional
import io
import re
import json

from datasets import load_dataset
from datasets.load import IterableDataset
from ray.data import from_huggingface, Dataset

from PIL import Image
from sycamore.context import Context
from sycamore.data.bbox import BoundingBox
from sycamore.data.document import Document
from sycamore.data.table import Table
from sycamore.docset import DocSet
from sycamore.plan_nodes import Scan


class TableEvalDoc(Document):
    def __init__(self, document=None, /, **kwargs):
        super().__init__(document, **kwargs)
        if document is None:
            self.data["type"] = "TableEvalDP"
            self.data["metrics"] = {}

    @property
    def gt_table(self) -> Optional[Table]:
        """Ground truth table"""
        return self.data.get("gt_table")

    @gt_table.setter
    def gt_table(self, table: Table) -> None:
        """Set the ground truth table"""
        self.data["gt_table"] = table

    @property
    def pred_table(self) -> Optional[Table]:
        """Predicted table"""
        return self.data.get("pred_table")

    @pred_table.setter
    def pred_table(self, table: Table) -> None:
        """Set the predicted table"""
        self.data["pred_table"] = table

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


class TableEvalScan(Scan):

    def format(self):  # type: ignore
        return "TableEval"

    def to_docset(self, context: Context) -> DocSet:
        return DocSet(context, self)


class PubTabNetScan(TableEvalScan):

    @staticmethod
    def _ray_row_to_document(row) -> dict[str, bytes]:
        if isinstance(row['image'], Image.Image):
            img = row['image']
        else:
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        table_pattern = r"<table[^>]*>.*?</table>"
        cleaning_pattern = r"<(?!/?(table|tr|td|thead|tbody)\b)[^>]+>"
        whitespace_removal = r"\s+"

        table_str = re.findall(table_pattern, row["html_table"], re.DOTALL)[0]
        table_str = re.sub(cleaning_pattern, "", table_str)
        table_str = re.sub(whitespace_removal, " ", table_str)
        table_str = re.sub(r"> ", ">", table_str)
        table_str = re.sub(r" <", "<", table_str)
        eval_doc = TableEvalDoc()
        eval_doc.image = img
        eval_doc.gt_table = Table.from_html(table_str)

        tokens = []
        cellsstr = (
            row["html"]
            .replace('''"'"''', "apostrophestupidness")
            .replace("\\xad", "-")
            .replace('"', '\\"')
            .replace("'", '"')
            .replace("apostrophestupidness", '''"'"''')
        )
        for cell in json.loads(cellsstr)["cells"]:
            if "bbox" not in cell or "tokens" not in cell:
                continue
            text = "".join(cell["tokens"])
            pattern = r"<[^>]+>"
            text = re.sub(pattern, "", text)
            bb = BoundingBox(
                x1=cell["bbox"][0] / img.width,
                y1=cell["bbox"][1] / img.height,
                x2=cell["bbox"][2] / img.width,
                y2=cell["bbox"][3] / img.height,
            )
            tokens.append({"text": text, "bbox": bb})
        eval_doc.properties["tokens"] = tokens

        return {"doc": eval_doc.serialize()}

    def execute(self, **kwargs) -> Dataset:
        hf_ds = load_dataset("apoidea/pubtabnet-html", split="validation", streaming=True)
        assert isinstance(hf_ds, IterableDataset)
        ray_ds = from_huggingface(hf_ds)
        return ray_ds.map(PubTabNetScan._ray_row_to_document)

<<<<<<< Updated upstream

class FinTabNetScan(TableEvalScan):

    def execute(self, **kwargs) -> Dataset:
        hf_ds = load_dataset("bsmock/FinTabNet.c", split="train", streaming=True)
        assert isinstance(hf_ds, IterableDataset)
        ray_ds = from_huggingface(hf_ds)
        return ray_ds.map(PubTabNetScan._ray_row_to_document)
=======
    def local_process(self, **kwargs) -> Iterable[Document]:
        hf_ds = load_dataset("apoidea/pubtabnet-html", split="validation", streaming=True)
        yield from (Document.deserialize(PubTabNetScan._ray_row_to_document(row)['doc']) for row in hf_ds)
>>>>>>> Stashed changes
