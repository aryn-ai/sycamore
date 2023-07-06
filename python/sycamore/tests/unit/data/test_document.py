from sycamore.data import (Document, Element)


class TestElement:
    def test_element(self):
        element = Element()
        assert (element.type is None)
        assert (element.content is None)
        assert (element.properties == {})

        element.type = "table"
        element.content = "text"
        element.properties.update({"property1": 1})
        assert (element.type == "table")
        assert (element.content == "text")
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
        assert (document.content is None)
        assert (document.elements == [])
        assert (document.embedding is None)
        assert (document.parent_id is None)
        assert (document.properties == {})

        document.doc_id = "doc_id"
        document.type = "table"
        document.content = "text"
        element = Element()
        document.elements.append(element)
        document.embedding = [[1.0, 2.0], [2.0, 3.0]]
        document.properties.update({"property1": 1})
        assert (document.doc_id == "doc_id")
        assert (document.type == "table")
        assert (document.content == "text")
        assert (document.elements == [element])
        assert (document.embedding == [[1.0, 2.0], [2.0, 3.0]])
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
