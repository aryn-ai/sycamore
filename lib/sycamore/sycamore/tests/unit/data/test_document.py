import pytest
import io
import struct
import msgpack

from sycamore.data import BoundingBox, Document, Element, MetadataDocument
from sycamore.data.element import TableElement
from sycamore.data.table import Table, TableCell
from sycamore.data.document import DOCUMENT_WEB_SERIALIZATION_HEADER_FORMAT


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

    def test_table_element_text(self):
        element = TableElement({"text_representation": "base text"})
        assert element.type == "table"
        assert element.text_representation == "base text"
        table = Table(
            [
                TableCell(content="1", rows=[0], cols=[0]),
                TableCell(content="2", rows=[0], cols=[1]),
                TableCell(content="3", rows=[1], cols=[0]),
                TableCell(content="4", rows=[1], cols=[1]),
            ]
        )
        element.table = table
        assert element.text_representation == table.to_csv()
        element.text_representation = "new text"
        assert element.text_representation == "new text"

        element.table = None
        assert element.table is None
        assert element.text_representation is None
        element.table = table
        assert element.text_representation == table.to_csv()


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

    def test_web_serialize_deserialize_basic(self):
        """Test basic web_serialize and web_deserialize functionality."""
        # Create a simple document
        doc = Document()
        doc.doc_id = "test_doc_123"
        doc.type = "pdf"
        doc.text_representation = "This is a test document"
        doc.binary_representation = b"binary content"
        doc.properties = {"author": "Test Author", "page_count": 5}

        # Serialize to bytes
        buffer = io.BytesIO()
        doc.web_serialize(buffer)
        buffer.seek(0)

        # Deserialize back
        deserialized_doc = Document.web_deserialize(buffer)

        # Verify all properties are preserved
        assert deserialized_doc.doc_id == "test_doc_123"
        assert deserialized_doc.type == "pdf"
        assert deserialized_doc.text_representation == "This is a test document"
        assert deserialized_doc.binary_representation == b"binary content"
        assert deserialized_doc.properties == {"author": "Test Author", "page_count": 5}
        assert len(deserialized_doc.elements) == 0

    def test_web_serialize_deserialize_with_elements(self):
        """Test web_serialize and web_deserialize with document elements."""
        # Create a document with elements
        doc = Document()
        doc.doc_id = "test_doc_with_elements"
        doc.type = "pdf"

        # Create elements
        element1 = Element()
        element1.type = "Text"
        element1.text_representation = "First element"
        element1.properties = {"font_size": 12}

        element2 = Element()
        element2.type = "Image"
        element2.binary_representation = b"image data"
        element2.properties = {"width": 100, "height": 200}

        doc.elements = [element1, element2]

        # Serialize to bytes
        buffer = io.BytesIO()
        doc.web_serialize(buffer)
        buffer.seek(0)

        # Deserialize back
        deserialized_doc = Document.web_deserialize(buffer)

        # Verify document properties
        assert deserialized_doc.doc_id == "test_doc_with_elements"
        assert deserialized_doc.type == "pdf"
        assert len(deserialized_doc.elements) == 2

        # Verify first element
        assert deserialized_doc.elements[0].type == "Text"
        assert deserialized_doc.elements[0].text_representation == "First element"
        assert deserialized_doc.elements[0].properties == {"font_size": 12}

        # Verify second element
        assert deserialized_doc.elements[1].type == "Image"
        assert deserialized_doc.elements[1].binary_representation == b"image data"
        assert set(deserialized_doc.elements[1].properties.keys()).issuperset({"width": 100, "height": 200})

    def test_web_serialize_deserialize_with_bbox(self):
        """Test web_serialize and web_deserialize with bounding box."""
        doc = Document()
        doc.doc_id = "test_doc_with_bbox"
        doc.bbox = BoundingBox(0.105, 0.203, 0.307, 0.401)

        # Serialize to bytes
        buffer = io.BytesIO()
        doc.web_serialize(buffer)
        buffer.seek(0)

        # Deserialize back
        deserialized_doc = Document.web_deserialize(buffer)

        # Verify bounding box is preserved
        assert deserialized_doc.bbox is not None
        assert deserialized_doc.bbox.coordinates == (0.105, 0.203, 0.307, 0.401)

    def test_web_serialize_deserialize_complex_properties(self):
        """Test web_serialize and web_deserialize with complex property types."""
        doc = Document()
        doc.doc_id = "test_doc_complex_props"
        doc.properties = {
            "string": "test string",
            "integer": 42,
            "float": 3.14159,
            "boolean": True,
            "list": [1, 2, 3, "four"],
            "dict": {"nested": "value", "numbers": [1, 2, 3]},
            "none": None,
        }

        # Serialize to bytes
        buffer = io.BytesIO()
        doc.web_serialize(buffer)
        buffer.seek(0)

        # Deserialize back
        deserialized_doc = Document.web_deserialize(buffer)

        # Verify complex properties are preserved
        assert deserialized_doc.properties["string"] == "test string"
        assert deserialized_doc.properties["integer"] == 42
        assert deserialized_doc.properties["float"] == 3.14159
        assert deserialized_doc.properties["boolean"] is True
        assert deserialized_doc.properties["list"] == [1, 2, 3, "four"]
        assert deserialized_doc.properties["dict"] == {"nested": "value", "numbers": [1, 2, 3]}
        assert deserialized_doc.properties["none"] is None

    def test_web_serialize_unsupported_document_types(self):
        """Test that web_serialize raises NotImplementedError for unsupported document types."""
        # Create a MetadataDocument (which is not supported)
        metadata_doc = MetadataDocument(test_prop="test_value")

        buffer = io.BytesIO()
        with pytest.raises(NotImplementedError, match="web_serialize cannot yet handle type 'MetadataDocument'"):
            metadata_doc.web_serialize(buffer)

    def test_web_deserialize_invalid_magic(self):
        """Test that web_deserialize raises RuntimeError for invalid magic bytes."""
        # Create invalid serialized data
        buffer = io.BytesIO()
        buffer.write(struct.pack(DOCUMENT_WEB_SERIALIZATION_HEADER_FORMAT, b"INVALID!", 0, 1))
        buffer.seek(0)

        with pytest.raises(
            RuntimeError, match=r"Input does not appear to be an Aryn serialized document \(Bad magic number\)\."
        ):
            Document.web_deserialize(buffer)

    def test_web_deserialize_unsupported_version(self):
        """Test that web_deserialize raises RuntimeError for unsupported versions."""
        # Create serialized data with unsupported version
        buffer = io.BytesIO()
        buffer.write(struct.pack(DOCUMENT_WEB_SERIALIZATION_HEADER_FORMAT, b"ArynSDoc", 65535, 65535))
        buffer.seek(0)

        with pytest.raises(RuntimeError, match="Unsupported serialization version: 65535.65535"):
            Document.web_deserialize(buffer)

    def test_web_deserialize_non_zero_padding(self):
        """Test that web_deserialize handles non-zero padding with a warning."""
        # Create serialized data with non-zero padding
        buffer = io.BytesIO()
        buffer.write(struct.pack("!8s2HI", b"ArynSDoc", 0, 1, 123))
        msgpack.pack({}, buffer)
        Element().web_serialize(buffer)
        msgpack.pack("_TERMINATOR", buffer)
        buffer.seek(0)

        # This should not raise an error
        Document.web_deserialize(buffer)

    def test_web_deserialize_terminator_missing(self):
        """Test that web_deserialize raises RuntimeError for missing terminator."""
        buffer = io.BytesIO()
        buffer.write(struct.pack(DOCUMENT_WEB_SERIALIZATION_HEADER_FORMAT, b"ArynSDoc", 0, 1))
        Element().web_serialize(buffer)
        buffer.seek(0)

        with pytest.raises(RuntimeError, match="Premature end of serialized document stream."):
            Document.web_deserialize(buffer)

    def test_web_serialize_deserialize_empty_document(self):
        """Test web_serialize and web_deserialize with minimal document."""
        doc = Document()

        # Serialize to bytes
        buffer = io.BytesIO()
        doc.web_serialize(buffer)
        buffer.seek(0)

        # Deserialize back
        deserialized_doc = Document.web_deserialize(buffer)

        # Verify basic structure is preserved
        assert deserialized_doc.doc_id is None
        assert deserialized_doc.type is None
        assert deserialized_doc.text_representation is None
        assert deserialized_doc.binary_representation is None
        assert deserialized_doc.elements == []
        assert deserialized_doc.properties == {}
        # lineage_id should be generated
        assert deserialized_doc.lineage_id is not None

    def test_web_serialize_deserialize_with_table_element(self):
        """Test web_serialize and web_deserialize with TableElement."""
        # Create a table
        table = Table(
            [
                TableCell(content="Header1", rows=[0], cols=[0], is_header=True),
                TableCell(content="Header2", rows=[0], cols=[1], is_header=True),
                TableCell(content="Data1", rows=[1], cols=[0], is_header=False),
                TableCell(content="Data2", rows=[1], cols=[1], is_header=False),
            ]
        )

        # Create document with table element
        doc = Document()
        doc.doc_id = "test_doc_with_table"

        table_element = TableElement()
        table_element.table = table
        table_element.properties = {"title": "Test Table"}

        doc.elements = [table_element]

        # Serialize to bytes
        buffer = io.BytesIO()
        doc.web_serialize(buffer)
        buffer.seek(0)

        # Deserialize back
        deserialized_doc = Document.web_deserialize(buffer)

        # Verify table element is preserved
        assert len(deserialized_doc.elements) == 1
        assert isinstance(deserialized_doc.elements[0], TableElement)
        assert deserialized_doc.elements[0].properties["title"] == "Test Table"
        assert deserialized_doc.elements[0].text_representation == "Header1,Header2\nData1,Data2\n"

    def test_web_serialize_deserialize_roundtrip_consistency(self):
        """Test that multiple serialize/deserialize cycles maintain consistency."""
        # Create a complex document
        doc = Document()
        doc.doc_id = "roundtrip_test"
        doc.type = "pdf"
        doc.text_representation = "Original text"
        doc.binary_representation = b"original binary"
        doc.embedding = [0.1, 0.2, 0.3]
        doc.properties = {"test": "value", "numbers": [1, 2, 3]}

        element = Element()
        element.type = "text"
        element.text_representation = "Element text"
        element.properties = {"element_prop": "element_value"}
        doc.elements = [element]

        # First roundtrip
        buffer1 = io.BytesIO()
        doc.web_serialize(buffer1)
        buffer1.seek(0)
        doc1 = Document.web_deserialize(buffer1)

        # Second roundtrip
        buffer2 = io.BytesIO()
        doc1.web_serialize(buffer2)
        buffer2.seek(0)
        doc2 = Document.web_deserialize(buffer2)

        # Verify consistency
        assert doc2.doc_id == "roundtrip_test"
        assert doc2.type == "pdf"
        assert doc2.text_representation == "Original text"
        assert doc2.binary_representation == b"original binary"
        assert doc2.embedding == [0.1, 0.2, 0.3]
        assert doc2.properties == {"test": "value", "numbers": [1, 2, 3]}
        assert len(doc2.elements) == 1
        assert doc2.elements[0].type == "text"
        assert doc2.elements[0].text_representation == "Element text"
        assert doc2.elements[0].properties == {"element_prop": "element_value"}


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
