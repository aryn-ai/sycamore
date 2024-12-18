import hashlib
from typing import cast
import boto3
import json
from sycamore.data.document import Document
from sycamore.data.element import TableElement
from sycamore.data.table import Table
from sycamore.evaluation.tables.benchmark_scans import TableEvalDoc
from sycamore.transforms import extract_table
from sycamore.transforms.table_structure.extract import TableStructureExtractor, TableTransformerStructureExtractor

from PIL import Image
import numpy as np
import re

from textractor import Textractor
from textractor.textractor import TextractFeatures
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrForObjectDetection


class ExtractTableFromImage:

    def __init__(self, extractor: TableStructureExtractor):
        self._extractor = extractor

    def extract_table(self, docs: list[Document]) -> list[Document]:
        ans = []
        for doc in docs:
            doc = TableEvalDoc(doc)
            assert isinstance(doc, TableEvalDoc), f"Wrong kind of doc: {type(doc)}, {doc}"
            image = doc.image
            assert image is not None
            table_bbox_element = TableElement(bbox=(0, 0, 1, 1), tokens=doc.properties["tokens"])
            predicted_elt = self._extractor.extract(table_bbox_element, image)
            if predicted_elt.table is not None:
                doc.pred_table = predicted_elt.data["table"]
            ans.append(cast(Document, doc))
        return ans

    def __call__(self, docs: list[Document]) -> list[Document]:
        return self.extract_table(docs)

class HomemadeTableTransformerTableStructureExtractor(TableTransformerStructureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.structure_model = DeformableDetrForObjectDetection.from_pretrained("/home/ubuntu/sycamore/tatr_real_3").to(self._get_device())

class PaddleTableStructureExtractor(TableStructureExtractor):

    def extract(self, element: TableElement, doc_image: Image.Image) -> TableElement:
        if not hasattr(self, "engine"):
            from paddleocr import PPStructure
            engine = PPStructure(lang='en', layout=False, show_log=False)
            self.engine = engine
        result = self.engine(np.array(doc_image))
        for elt in result:
            if elt['type'] == 'table':
                table_pattern = r"<table[^>]*>.*?</table>"
                table_str = re.findall(table_pattern, elt['res']["html"], re.DOTALL)[0]
                element.table = Table.from_html(table_str)
                return element
        raise RuntimeError(f"Did not find a table: results: {result}")

class PaddleV2TableStructureExtractor(TableStructureExtractor):

    def extract(self, element: TableElement, doc_image: Image.Image) -> TableElement:
        if not hasattr(self, "engine"):
            from paddleocr import PPStructure
            engine = PPStructure(recovery=True, structure_version="PP-StructureV2", layout=False, return_ocr_result_in_table=True, show_log=False)
            self.engine = engine
        result = self.engine(np.array(doc_image))
        for elt in result:
            if elt['type'] == 'table':
                table_pattern = r"<table[^>]*>.*?</table>"
                table_str = re.findall(table_pattern, elt['res']["html"], re.DOTALL)[0]
                element.table = Table.from_html(table_str)
                return element
        raise RuntimeError(f"Did not find a table: results: {result}")


class TextractTableStructureExtractor(TableStructureExtractor):

    S3_BUCKET = 'henry-textract'
    CACHE_PREFIX = 'cache/'

    @staticmethod
    def _response_to_table_element(resp) -> TableElement:
        tables = resp.tables
        if len(tables) == 0:
            print(resp)
            # If no table, make a null 1-cell table
            table = Table.from_html("<table><tr><td /></tr></table>")
        else:
            table = Table.from_html(resp.tables[0].to_html())
        return TableElement(table=table)

    def extract(self, element: TableElement, doc_image: Image.Image) -> TableElement:
        import hashlib
        import boto3
        from textractor.parsers import response_parser
        from botocore.exceptions import ClientError

        # Cache lookup
        hasher = hashlib.sha1()
        hasher.update(doc_image.tobytes())
        key = hasher.hexdigest()
        s3 = boto3.client("s3")
        try:
            response = s3.get_object(Bucket = self.S3_BUCKET, Key = self.CACHE_PREFIX + key)
            parsed_response = response_parser.parse(json.loads(response['Body'].read()))
            return self._response_to_table_element(parsed_response)
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchKey':
                raise e

        # Cache miss
        from textractor import Textractor
        from textractor.data.constants import TextractFeatures

        extractor = Textractor(profile_name='admin')
        doc = extractor.analyze_document(doc_image, TextractFeatures.TABLES)
        s3.put_object(Body=json.dumps(doc.response), Bucket=self.S3_BUCKET, Key = self.CACHE_PREFIX + key)
        return self._response_to_table_element(doc)


class FlorenceTableStructureExtractor(TableStructureExtractor):

    MODEL_ID = "ucsahin/Florence-2-large-TableDetection"
    MODEL_ID = "microsoft/Florence-2-large-ft"

    def __init__(self):
        from transformers import AutoProcessor, AutoModelForCausalLM

        self._model = AutoModelForCausalLM.from_pretrained(
            FlorenceTableStructureExtractor.MODEL_ID,
            trust_remote_code = True,
            device_map = "cuda",
        )
        self._processor = AutoProcessor.from_pretrained(
            FlorenceTableStructureExtractor.MODEL_ID,
            trust_remote_code = True,
        )

    def extract(self, element: TableElement, doc_image: Image.Image) -> TableElement:
        prompt = "How many columns are in this table?"
        inputs = self._processor(text=prompt, images=doc_image, return_tensors="pt")
        gen_ids = self._model.generate(
            input_ids = inputs['input_ids'].cuda(),
            pixel_values = inputs['pixel_values'].cuda(),
            num_beams = 3,
        )
        gen_text = self._processor.batch_decode(gen_ids, skip_special_tokens = False)[0]
        print(f">> {gen_text}")
        parsed_answer = self._processor.post_process_generation(gen_text, task=prompt, image_size=(doc_image.width, doc_image.height))
        print(f">>>> {parsed_answer}\n")
        if "<table" in gen_text:
            table_pattern = r"<table[^>]*>.*?</table>"
            table_str = re.findall(table_pattern, gen_text, re.DOTALL)[0]
            element.table = Table.from_html(table_str)
        else:
            element.table = Table.from_html("<table><tr><td /></tr></table>")
        return element
