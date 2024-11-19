from typing import Iterable, Optional
import io
import re
import json
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import logging

from datasets import load_dataset
from datasets.load import IterableDataset
from ray.data import from_huggingface, Dataset

from PIL import Image
import pdf2image
from sycamore.context import Context
from sycamore.data.bbox import BoundingBox
from sycamore.data.document import Document
from sycamore.data.table import Table, TableCell
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
        if isinstance(row["image"], Image.Image):
            img = row["image"]
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

    def local_process(self, **kwargs) -> Iterable[Document]:
        hf_ds = load_dataset("apoidea/pubtabnet-html", split="validation", streaming=True)
        yield from (Document.deserialize(PubTabNetScan._ray_row_to_document(row)["doc"]) for row in hf_ds)


class FinTabNetS3Scan(TableEvalScan):

    FINTABNET_S3_BUCKET = "aryn-datasets-us-east-1"
    FINTABNET_ANNOTATIONS_KEY = "fintabnet/FinTabNet.c.jsonl"
    FINTABNET_PDF_PREFIX = "fintabnet/fintabnet/pdf/"

    @staticmethod
    def _json_object_to_document(obj):
        cells = []
        tokens = []
        for c in obj["cells"]:
            bb = BoundingBox(
                x1=c["pdf_bbox"][0],
                y1=c["pdf_bbox"][1],
                x2=c["pdf_bbox"][2],
                y2=c["pdf_bbox"][3],
            )
            tc = TableCell(
                content=c["pdf_text_content"],
                rows=c["row_nums"],
                cols=c["column_nums"],
                is_header=c["is_column_header"],
                bbox=bb,
            )
            cells.append(tc)
            tokens.append({"text": c["pdf_text_content"], "bbox": bb})
        table = Table(cells=cells)
        file = obj["pdf_folder"] + obj["pdf_file_name"]
        bbox = BoundingBox(*obj["pdf_table_bbox"])
        ed = TableEvalDoc()
        ed.gt_table = table
        ed.bbox = bbox
        ed.properties["path"] = file
        ed.properties["tokens"] = tokens
        page_bbox = BoundingBox(*obj["pdf_full_page_bbox"])
        ed.properties["desired_page_dimensions"] = int(page_bbox.width), int(page_bbox.height)
        return ed

    @staticmethod
    def _load_images(s3):

        def inner_load_image(doc):
            pdf_key = FinTabNetS3Scan.FINTABNET_PDF_PREFIX + doc.properties["path"]
            object = s3.get_object(Bucket=FinTabNetS3Scan.FINTABNET_S3_BUCKET, Key=pdf_key)
            byteses = object["Body"].read()
            images = pdf2image.convert_from_bytes(byteses)
            ed = TableEvalDoc(doc)
            ed.image = images[0]
            return ed

        return inner_load_image

    @staticmethod
    def _scaling(doc):
        ed = TableEvalDoc(doc)
        left, top = ed.bbox.x1, ed.bbox.y1
        im = ed.image
        im = im.resize(ed.properties["desired_page_dimensions"])
        ed.image = im.crop(box=ed.bbox.coordinates)
        # ed.bbox = ed.bbox.to_relative(full_width, full_height)
        for tk in ed.properties["tokens"]:
            tk["bbox"] = tk["bbox"].translate_self(-left, -top).to_relative_self(ed.bbox.width, ed.bbox.height)
        return ed

    @staticmethod
    def _limit(n):
        i = 0
        def limit_predicate(doc):
            nonlocal i
            i += 1
            return i < n or n == -1
        return limit_predicate

    @staticmethod
    def _sample(factor):
        def sample_predicate(doc):
            return random.random() < factor
        return sample_predicate

    def execute(self, **kwargs) -> Dataset:
        pass

    def local_process(self, **kwargs) -> Iterable[Document]:
        import boto3

        s3 = boto3.client("s3")
        annotations_response = s3.get_object(
            Bucket=FinTabNetS3Scan.FINTABNET_S3_BUCKET, Key=FinTabNetS3Scan.FINTABNET_ANNOTATIONS_KEY
        )
        limit = kwargs.get('limit', -1)
        sample_factor = kwargs.get('sample_factor', 1)

        lines_stream = annotations_response['Body'].iter_lines()
        sample_stream = filter(FinTabNetS3Scan._sample(sample_factor), lines_stream)
        limited_stream = filter(FinTabNetS3Scan._limit(limit), sample_stream)
        json_stream = map(json.loads, limited_stream)
        json_object_stream = map(lambda o: o[0], json_stream)
        document_stream = map(FinTabNetS3Scan._json_object_to_document, json_object_stream)
        imaged_stream = map(FinTabNetS3Scan._load_images(s3), document_stream)
        scaled_stream = map(FinTabNetS3Scan._scaling, imaged_stream)
        yield from scaled_stream


class CohereTabNetS3Scan(TableEvalScan):
    COHERETABNET_S3_BUCKET = "aryn-datasets-us-east-1"
    COHERETABNET_ANNOTATIONS_KEY = "coheretabnet/annotations.jsonl"
    COHERETABNET_PDF_PREFIX = "coheretabnet/pdf/"

    @staticmethod
    def _row_to_doc(row) -> Document:
        table = Table.from_html(row['html'])
        bbox = BoundingBox(*row['table_bbox'])
        left, top = bbox.x1, bbox.y1
        tokens = []
        for cell in row['cells']:
            bb = BoundingBox(*cell['bbox']).translate(-left, -top).to_relative(bbox.width, bbox.height)
            tokens.append({
                "text": cell["text"].strip(),
                "bbox": bb,
            })
        path = CohereTabNetS3Scan.COHERETABNET_PDF_PREFIX + row['pdf_page']
        ed = TableEvalDoc()
        ed.gt_table = table
        ed.bbox = bbox
        ed.properties['tokens'] = tokens
        ed.properties['path'] = path
        return ed

    @staticmethod
    def _load_images(s3):

        def inner_load_image(doc):
            pdf_key = doc.properties["path"]
            object = s3.get_object(Bucket=CohereTabNetS3Scan.COHERETABNET_S3_BUCKET, Key=pdf_key)
            byteses = object["Body"].read()
            images = pdf2image.convert_from_bytes(byteses)
            ed = TableEvalDoc(doc)
            ed.image = images[0].crop(tuple([int(coord) for coord in ed.bbox.coordinates]))
            return ed

        return inner_load_image

    def local_process(self, **kwargs) -> Iterable[Document]:
        import boto3

        s3 = boto3.client("s3")
        annotations_response = s3.get_object(
            Bucket = CohereTabNetS3Scan.COHERETABNET_S3_BUCKET,
            Key = CohereTabNetS3Scan.COHERETABNET_ANNOTATIONS_KEY
        )
        limit = kwargs.get('limit', -1)
        sample_factor = kwargs.get('sample_factor', 1)

        lines_stream = annotations_response['Body'].iter_lines()
        sample_stream = filter(FinTabNetS3Scan._sample(sample_factor), lines_stream)
        limited_stream = filter(FinTabNetS3Scan._limit(limit), sample_stream)
        json_stream = map(json.loads, limited_stream)
        document_stream = map(CohereTabNetS3Scan._row_to_doc, json_stream)
        imaged_stream = map(CohereTabNetS3Scan._load_images(s3), document_stream)
        yield from imaged_stream

    def execute(self, **kwargs) -> "Dataset":
        pass

class IcdarFileScan(TableEvalScan):
    LOCAL_PATH = "/home/ubuntu/datasets/icdar/ICDAR-2013.c-Structure"

    @staticmethod
    def _read_pascal_voc(xml_file: Path, class_map={
        "table": 0,
        "table column": 1,
        "table column header": 3,
        "table projected row header": 4,
        "table row": 2,
        "table spanning cell": 5, "no object": 6
    }):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as e:
            logging.error(f"Error parsing {xml_file}")
            raise e
        root = tree.getroot()

        bboxes = []
        labels = []

        for object_ in root.iter("object"):
            ymin, xmin, ymax, xmax = None, None, None, None

            label = object_.find("name").text
            # TODO: Fix typos in the data, not code
            try:
                label = int(label)
            except:
                label = int(class_map[label])

            for box in object_.findall("bndbox"):
                ymin = float(box.find("ymin").text)
                xmin = float(box.find("xmin").text)
                ymax = float(box.find("ymax").text)
                xmax = float(box.find("xmax").text)

            bbox = [xmin, ymin, xmax, ymax]  # PASCAL VOC

            bboxes.append(bbox)
            labels.append(label)

        return xml_file, bboxes, labels

    @staticmethod
    def _read_words_json(xml_file):
        json_file = Path(str(xml_file).replace(".xml", "_words.json").replace("test", "words"))
        with open(json_file, "r") as f:
            words = json.load(f)


    def local_process(self, **kwargs):
        test_path = Path(IcdarFileScan.LOCAL_PATH) / "test"
        f_iterator = test_path.iterdir()
        fileboxesandlabels = map(IcdarFileScan._read_pascal_voc, f_iterator)
