from sycamore.data import Element
from sycamore.transforms.detr_partitioner import ArynPDFPartitioner, DeformableDetr
from sycamore.data import BoundingBox
from sycamore.tests.unit.transforms.compare_detr_impls import compare_batched_sequenced, check_table_extraction

from PIL import Image
import json
from sycamore.tests.config import TEST_DIR


class TestArynPDFPartitioner:
    def test_supplement_text(self):
        infer1 = Element()
        infer2 = Element()
        infer3 = Element()
        infer1.bbox = BoundingBox(20, 20, 100, 100)
        infer2.bbox = BoundingBox(60, 10, 140, 60)
        infer3.bbox = BoundingBox(20, 120, 100, 200)

        miner1 = Element()
        miner2 = Element()
        miner3 = Element()
        miner4 = Element()
        miner5 = Element()
        miner6 = Element()
        miner1.text_representation = "hello, world"
        miner1.bbox = BoundingBox(21, 21, 59, 59)
        miner2.text_representation = "你好，世界"
        miner2.bbox = BoundingBox(61, 21, 99, 59)
        miner3.text_representation = "Bonjour le monde"
        miner3.bbox = BoundingBox(21, 71, 99, 99)
        miner4.text_representation = "Hola Mundo"
        miner4.bbox = BoundingBox(25, 125, 105, 205)
        miner5.text_representation = "Ciao mondo"
        miner5.bbox = BoundingBox(25, 250, 100, 300)
        miner6.bbox = BoundingBox(21, 71, 99, 99)

        result = ArynPDFPartitioner._supplement_text(
            [infer1, infer2, infer3], [miner1, miner2, miner3, miner4, miner5, miner6]
        )
        assert result[0].text_representation == "hello, world 你好，世界 Bonjour le monde"
        assert result[1].text_representation == "你好，世界"
        assert result[2].text_representation == "Hola Mundo"
        assert result[3].text_representation == "Ciao mondo"

    def test_infer(self):
        with Image.open(TEST_DIR / "resources/data/imgs/sample-detr-image.png") as image:
            d = DeformableDetr("Aryn/deformable-detr-DocLayNet")
            results = d.infer([image], 0.7)

            for result in results:
                for element in result:
                    json.dumps(element.properties)

    def test_batched_sequenced(self):
        s = ArynPDFPartitioner("Aryn/deformable-detr-DocLayNet")
        d = compare_batched_sequenced(
            s, TEST_DIR / "../../../../apps/crawler/crawler/http/tests/visit_aryn.pdf", use_cache=False
        )
        assert len(d) == 1
        d = compare_batched_sequenced(s, TEST_DIR / "resources/data/pdfs/basic_table.pdf", use_cache=False)
        assert len(d) == 1
        d = compare_batched_sequenced(
            s, TEST_DIR / "resources/data/pdfs/basic_table.pdf", use_ocr=True, use_cache=False
        )
        assert len(d) == 1

    def test_table_extraction_order(self):
        s = ArynPDFPartitioner("Aryn/deformable-detr-DocLayNet")
        d = check_table_extraction(
            s,
            TEST_DIR / "../../../../apps/crawler/crawler/http/tests/visit_aryn.pdf",
            extract_table_structure=True,
            use_cache=False,
        )
        assert len(d) == 1
        d = check_table_extraction(
            s, TEST_DIR / "resources/data/pdfs/basic_table.pdf", extract_table_structure=True, use_cache=False
        )
        assert len(d) == 1
        d = check_table_extraction(
            s,
            TEST_DIR / "resources/data/pdfs/basic_table.pdf",
            extract_table_structure=True,
            use_ocr=True,
            use_cache=False,
        )
        assert len(d) == 1
