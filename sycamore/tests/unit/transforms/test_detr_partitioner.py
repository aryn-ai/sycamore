from sycamore.data import Element
from sycamore.transforms import SycamorePDFPartitioner
from sycamore.data import BoundingBox


class TestSycamorePDFPartitioner:
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

        result = SycamorePDFPartitioner._supplement_text(
            [infer1, infer2, infer3], [miner1, miner2, miner3, miner4, miner5, miner6]
        )
        assert result[0].text_representation == "hello, world 你好，世界 Bonjour le monde"
        assert result[1].text_representation == "你好，世界"
        assert result[2].text_representation == "Hola Mundo"
        assert result[3].text_representation == "Ciao mondo"
