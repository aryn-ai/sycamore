from unittest.mock import Mock

from sycamore.data import Element
from sycamore.transforms.detr_partitioner import ArynPDFPartitioner, DeformableDetr
from sycamore.data import BoundingBox
from sycamore.tests.unit.transforms.check_partition_impl import check_partition, check_table_extraction
from sycamore.transforms.text_extraction import get_text_extractor, PdfMinerExtractor

from PIL import Image
import json
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.text_extraction import OcrModel


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

    def test_partition(self):
        s = ArynPDFPartitioner("Aryn/deformable-detr-DocLayNet")
        d = check_partition(s, TEST_DIR / "resources/data/pdfs/visit_aryn.pdf", use_cache=False)
        assert len(d) == 1
        d = check_partition(s, TEST_DIR / "resources/data/pdfs/basic_table.pdf", use_cache=False)
        assert len(d) == 1
        d = check_partition(s, TEST_DIR / "resources/data/pdfs/basic_table.pdf", use_ocr=True, use_cache=False)
        assert len(d) == 1

    def test_partition_w_ocr_instance(self):
        s = ArynPDFPartitioner("Aryn/deformable-detr-DocLayNet")
        ocr = Mock(spec=OcrModel)
        dummy_text = "mocked ocr text"
        ocr.get_text.return_value = dummy_text
        d = check_partition(
            s, TEST_DIR / "resources/data/pdfs/visit_aryn.pdf", use_ocr=True, use_cache=False, ocr_model=ocr
        )
        assert len(d) == 1
        assert d[0][0].text_representation == dummy_text

    def test_table_extraction_order(self):
        # In non-ocr mode partitioning basic_table.pdf will fail to include the table output in the correct format
        # (with the bounding box as a list), and instead return it as a BoundingBox type.
        # This indicates that the table extraction model never saw the input tokens to help with extraction,
        # and hence was not able to modify it to the correct format.
        s = ArynPDFPartitioner("Aryn/deformable-detr-DocLayNet")
        d = check_table_extraction(
            s,
            TEST_DIR / "resources/data/pdfs/visit_aryn.pdf",
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

    def test_pdfminer_object_type(self):
        filename = str(TEST_DIR / "resources/data/pdfs/Ray_page11.pdf")
        lines_extractor = get_text_extractor("pdfminer", object_type="lines")
        pages = PdfMinerExtractor.pdf_to_pages(file_name=filename)

        lines_elements = []
        for i, p in enumerate(pages):
            assert i == 0
            lines_elements.extend(lines_extractor.extract_page(p))

        objects_extractor = get_text_extractor("pdfminer", object_type="boxes")
        pages = PdfMinerExtractor.pdf_to_pages(file_name=filename)

        objects_elements = []
        for i, p in enumerate(pages):
            assert i == 0
            objects_elements.extend(objects_extractor.extract_page(p))

        # Note: It's possible that for some documents these values would be equal, but
        # I was worried that if I used <= I might silently mask an issue where we weren't
        # actually honoring the object_type parameter. For this doc the values are quite
        # different
        assert len(objects_elements) < len(lines_elements)
        print(f"objects_elements {len(objects_elements)}, lines_elements {len(lines_elements)}")

        lines_text = "".join([e.text_representation for e in lines_elements if e.text_representation is not None])
        objects_text = "".join([e.text_representation for e in objects_elements if e.text_representation is not None])

        # I was very surprised that this equality succeeded. I'm not sure in general we can expect
        # exact text equality. I imagine in some cases the order might be different, but in this case
        # they match, so I'm asserting here so we can catch regressions.
        assert lines_text == objects_text

        print(f"objects_text {len(objects_text)}, lines_text {len(lines_text)}")

    def test_detr_pdfminer_object_type(self):
        s = ArynPDFPartitioner("Aryn/deformable-detr-DocLayNet")
        lines_result = check_table_extraction(
            s,
            TEST_DIR / "resources/data/pdfs/Ray_page11.pdf",
            extract_table_structure=True,
            use_cache=False,
            text_extraction_options={"object_type": "lines"},
        )

        assert len(lines_result) == 1
        lines_page = lines_result[0]

        objects_result = check_table_extraction(
            s,
            TEST_DIR / "resources/data/pdfs/Ray_page11.pdf",
            extract_table_structure=True,
            use_cache=False,
            text_extraction_options={"object_type": "objects"},
        )

        assert len(objects_result) == 1
        objects_page = objects_result[0]

        assert len(lines_page) == len(objects_page)

        lines_text = "".join(el.text_representation for el in lines_page if el.text_representation is not None)
        objects_text = "".join(el.text_representation for el in objects_page if el.text_representation is not None)

        assert lines_text == objects_text
