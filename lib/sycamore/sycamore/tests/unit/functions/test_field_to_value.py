from sycamore.data.document import Document

class TestFieldToValue:
    def test_field_to_value(self):
        doc = Document(
            text_representation="hello",
            doc_id=1,
            properties={"letter": "A", "animal": "panda", "math": {"pi": 3.14, "e": 2.72, "tanx": "sinx/cosx"}},
        )

        assert doc.field_to_value("text_representation") == "hello"
        assert doc.field_to_value("doc_id") == 1
        assert doc.field_to_value("properties.letter") == "A"
        assert doc.field_to_value("properties.animal") == "panda"
        assert doc.field_to_value("properties.math.pi") == 3.14
        assert doc.field_to_value("properties.math.e") == 2.72
        assert doc.field_to_value("properties.math.tanx") == "sinx/cosx"

        assert doc.field_to_value("properties.math.log") is None
        assert doc.field_to_value("document_id") is None
        assert doc.field_to_value("text_representation.text") is None
        assert doc.field_to_value("document_id.text") is None