import pytest

from sycamore.data import BoundingBox, Document, Element, MetadataDocument


class TestElement:
    def test_element(self):
        element = Element()
        assert element.type is None
        assert element.text_representation is None
        assert element.binary_representation is None
        assert element.bbox is None
        assert element.properties == {}

        element.type = "table"
        element.text_representation = "text"
        element.bbox = BoundingBox(1, 2, 3, 4)
        element.properties.update({"property1": 1})
        assert element.type == "table"
        assert element.text_representation == "text"
        assert element.properties == {"property1": 1}

        element.properties.update({"property2": 2})
        assert element.properties == {"property1": 1, "property2": 2}

        del element.properties
        assert element.properties == {}
        assert element.data["bbox"] == (1, 2, 3, 4)


class TestDocument:
    def test_document(self):
        document = Document()
        assert document.doc_id is None
        assert document.type is None
        assert document.text_representation is None
        assert document.binary_representation is None
        assert document.elements == []
        assert document.embedding is None
        assert document.parent_id is None
        assert document.bbox is None
        assert document.properties == {}

        document.doc_id = "doc_id"
        document.type = "table"
        document.text_representation = "text"
        element1 = Element()
        document.elements = [element1]
        document.embedding = [[1.0, 2.0], [2.0, 3.0]]
        document.bbox = BoundingBox(1, 2, 3, 4)
        document.properties["property1"] = 1
        assert document.doc_id == "doc_id"
        assert document.type == "table"
        assert document.text_representation == "text"
        assert document.elements == [element1.data]
        assert document.embedding == [[1.0, 2.0], [2.0, 3.0]]
        assert document.properties == {"property1": 1}
        document.properties = {"property2": 2}
        assert len(document.properties) == 1
        assert document.properties == {"property2": 2}

        element2 = Element({"type": "image", "text": "text"})
        document.elements = [element1, element2]
        assert document.elements == [element1.data, element2.data]
        document.properties["property3"] = 3
        document.properties.update({"property4": 4})
        assert document.properties == {"property2": 2, "property3": 3, "property4": 4}

        del document.elements
        del document.properties
        assert document.elements == []
        assert document.properties == {}

        assert document.data["bbox"] == (1, 2, 3, 4)

    def test_serde(self):
        dict = {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "text_representation",
            "bbox": (1, 2.3, 3.4, 4.5),
            "elements": [
                {
                    "type": "table",
                    "bbox": (1, 2, 3, 4.0),
                    "properties": {"title": None, "rows": None, "columns": None},
                    "table": None,
                    "tokens": None,
                },
                {
                    "type": "Image",
                    "bbox": (1, 2, 3, 4.0),
                    "properties": {"image_size": None, "image_mode": None, "image_format": None},
                },
            ],
            "properties": {"int": 0, "float": 3.14, "list": [1, 2, 3, 4], "tuple": (1, "tuple")},
        }
        document = Document(dict)
        serde = Document(document.serialize())
        print(serde.data)

        assert "lineage_id" in serde.data
        del serde.data["lineage_id"]
        assert serde.data == dict

    def test_element_typechecking(self):
        with pytest.raises(ValueError):
            Document({"elements": {}})
        with pytest.raises(ValueError):
            Document({"elements": ["abc"]})
        Document({"elements": [{"type": "special"}]})

    def test_elements_does_not_copy(self):
        d = Document({"elements": [{"type": "b"}, {"type": "a"}]})
        assert d.elements[0]["type"] == "b"
        assert d.elements[1]["type"] == "a"
        d.elements.sort(key=lambda v: v["type"])
        assert d.elements[0]["type"] == "a"
        assert d.elements[1]["type"] == "b"


class TestMetadataDocument:
    def test_fail_constructor(self):
        md = MetadataDocument(nobel="curie")
        ser = md.serialize()
        with pytest.raises(ValueError):
            Document(ser)

    def test_becomes_md_doc(self):
        md = MetadataDocument(nobel="curie")
        assert md.metadata["nobel"] == "curie"
        ser = md.serialize()
        d = Document.deserialize(ser)
        assert isinstance(d, MetadataDocument)
        assert isinstance(d, Document)
        assert d.metadata["nobel"] == "curie"

    def test_document_remains_unchanged(self):
        orig = Document(nobel="curie")
        d = Document.deserialize(orig.serialize())
        assert not isinstance(d, MetadataDocument)
        assert isinstance(d, Document)
        assert orig.lineage_id == d.lineage_id
