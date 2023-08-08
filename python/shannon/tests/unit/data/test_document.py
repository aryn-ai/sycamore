from shannon.data import (Document, Element)


class TestElement:
    def test_element(self):
        element = Element()
        assert (element.type is None)
        assert (element.content == {"binary": None, "text": None})
        assert (element.properties == {})

        element.type = "table"
        element.content.update({"text": "text"})
        element.properties.update({"property1": 1})
        assert (element.type == "table")
        assert (element.content == {"binary": None, "text": "text"})
        assert (element.properties == {"property1": 1})

        element.properties.update({"property2": 2})
        assert (element.properties == {"property1": 1, "property2": 2})

        del element.properties
        assert (element.properties == {})


class TestDocument:
    def test_document(self):
        document = Document()
        assert (document.doc_id is None)
        assert (document.type is None)
        assert (document.content == {"binary": None, "text": None})
        assert (document.elements == [])
        assert (document.embedding == {"binary": None, "text": None})
        assert (document.parent_id is None)
        assert (document.properties == {})

        document.doc_id = "doc_id"
        document.type = "table"
        document.content.update({"text": "text"})
        element = Element()
        document.elements.append(element)
        document.embedding.update({"text": [[1.0, 2.0], [2.0, 3.0]]})
        document.properties.update({"property1": 1})
        assert (document.doc_id == "doc_id")
        assert (document.type == "table")
        assert (document.content == {"binary": None, "text": "text"})
        assert (document.elements == [element])
        assert (document.embedding ==
                {"binary": None, "text": [[1.0, 2.0], [2.0, 3.0]]})
        assert (document.properties == {"property1": 1})

        element2 = Element({"type": "image", "text": "text"})
        document.elements.append(element2)
        assert (document.elements == [element, element2])
        document.properties.update({"property2": 2})
        assert (document.properties == {"property1": 1, "property2": 2})

        del document.elements
        del document.properties
        assert (document.elements == [])
        assert (document.properties == {})
