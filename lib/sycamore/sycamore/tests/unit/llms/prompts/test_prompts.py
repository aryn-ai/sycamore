from sycamore.data.element import Element
from sycamore.llms.prompts.prompts import (
    RenderedPrompt,
    RenderedMessage,
    StaticPrompt,
    SycamorePrompt,
    ElementPrompt,
    ElementListPrompt,
)
from sycamore.data import Document
from sycamore.tests.config import TEST_DIR
from pyarrow.fs import LocalFileSystem
import pytest


@pytest.fixture(scope="module")
def dummy_document():
    docpath = TEST_DIR / "resources/data/pdfs/ntsb-report.pdf"
    local = LocalFileSystem()
    path = str(docpath)
    input_stream = local.open_input_stream(path)
    document = Document()
    document.binary_representation = input_stream.readall()
    document.type = "pdf"
    document.properties["path"] = path
    document.properties["pages"] = 6
    document.elements = [
        Element(
            text_representation="Element 1",
            type="Text",
            element_id="e1",
            properties={"page_number": 1},
            bbox=(0.1, 0.1, 0.4, 0.4),
        ),
        Element(
            text_representation="Element 2",
            type="Text",
            element_id="e2",
            properties={"page_number": 2},
            bbox=(0.1, 0.1, 0.4, 0.4),
        ),
        Element(
            text_representation="Element 3",
            type="Text",
            element_id="e3",
            properties={"page_number": 3},
            bbox=(0.1, 0.1, 0.4, 0.4),
        ),
        Element(
            text_representation="Element 4",
            type="Text",
            element_id="e4",
            properties={"page_number": 3},
            bbox=(0.4, 0.1, 0.8, 0.4),
        ),
        Element(
            text_representation="Element 5",
            type="Text",
            element_id="e5",
            properties={"page_number": 3},
            bbox=(0.1, 0.4, 0.8, 0.8),
        ),
        Element(
            text_representation="Element 6",
            type="Text",
            element_id="e6",
            properties={"page_number": 4},
            bbox=(0.1, 0.1, 0.4, 0.4),
        ),
    ]
    return document


class TestRenderedPrompt:
    """RenderedPrompt and RenderedMessage are dataclasses,
    no need to test them. Nothing to test :)"""

    pass


class TestSycamorePrompt:
    def test_set_is_cow(self):
        sp = SycamorePrompt()
        sp.__dict__["key"] = "value"
        sp2 = sp.fork(key="other value")
        assert sp.key == "value"
        assert sp2.key == "other value"


class TestStaticPrompt:
    def test_static_rd(self, dummy_document):
        prompt = StaticPrompt(system="system {x}", user="computers")
        with pytest.raises(KeyError):
            prompt.render_document(dummy_document)

        prompt = prompt.fork(x=76)
        expected = RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="system 76"),
                RenderedMessage(role="user", content="computers"),
            ]
        )
        assert prompt.render_document(dummy_document) == expected
        assert prompt.render_element(dummy_document.elements[0], dummy_document) == expected
        assert prompt.render_multiple_documents([dummy_document]) == expected

    def test_static_with_multiple_user_prompts(self, dummy_document):
        prompt = StaticPrompt(system="system {x}", user=["{x} user {y}", "{x} user {z}"], x=1, y=2, z=3)
        expected = RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="system 1"),
                RenderedMessage(role="user", content="1 user 2"),
                RenderedMessage(role="user", content="1 user 3"),
            ]
        )
        assert prompt.render_document(dummy_document) == expected


class TestElementPrompt:
    def test_basic(self, dummy_document):
        prompt = ElementPrompt(
            system="You know everything there is to know about jazz, {name}",
            user="Summarize the information on page {elt_property_page_number}.\nTEXT: {elt_text}",
            name="Frank Sinatra",
        )
        expected = RenderedPrompt(
            messages=[
                RenderedMessage(
                    role="system", content="You know everything there is to know about jazz, Frank Sinatra"
                ),
                RenderedMessage(role="user", content="Summarize the information on page 3.\nTEXT: Element 4"),
            ]
        )
        assert prompt.render_element(dummy_document.elements[3], dummy_document) == expected
        with pytest.raises(NotImplementedError):
            prompt.render_document(dummy_document)
        with pytest.raises(NotImplementedError):
            prompt.render_multiple_documents([dummy_document])

    def test_get_parent_context(self, dummy_document):
        prompt = ElementPrompt(
            system="You know everything there is to know about {custom_property}, {name}",
            user="Summarize the information on page {elt_property_page_number}.\nTEXT: {elt_text}",
            name="Frank Sinatra",
            capture_parent_context=lambda doc, elt: {"custom_property": doc.properties["pages"]},
        )
        expected = RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="You know everything there is to know about 6, Frank Sinatra"),
                RenderedMessage(role="user", content="Summarize the information on page 3.\nTEXT: Element 4"),
            ]
        )
        assert prompt.render_element(dummy_document.elements[3], dummy_document) == expected

    def test_include_image(self, dummy_document):
        prompt = ElementPrompt(
            system="You know everything there is to know about {custom_property}, {name}",
            user="Summarize the information on page {elt_property_page_number}.\nTEXT: {elt_text}",
            name="Frank Sinatra",
            capture_parent_context=lambda doc, elt: {"custom_property": doc.properties["pages"]},
            include_element_image=True,
        )
        rp = prompt.render_element(dummy_document.elements[3], dummy_document)
        assert rp.messages[1].images is not None and len(rp.messages[1].images) == 1
        assert rp.messages[1].role == "user"
        assert rp.messages[0].images is None

        prompt = prompt.fork(user=None)
        rp2 = prompt.render_element(dummy_document.elements[1], dummy_document)
        assert len(rp2.messages) == 1
        assert rp2.messages[0].role == "system"
        assert rp2.messages[0].images is not None
        assert len(rp2.messages[0].images) == 1


class TestElementListPrompt:
    def test_basic(self, dummy_document):
        prompt = ElementListPrompt(system="sys", user="usr: {elements}")
        expected = RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="sys"),
                RenderedMessage(
                    role="user",
                    content="usr: ELEMENT 0: Element 1\nELEMENT 1: Element 2\n"
                    "ELEMENT 2: Element 3\nELEMENT 3: Element 4\nELEMENT 4: Element 5\nELEMENT 5: Element 6",
                ),
            ]
        )
        assert prompt.render_document(dummy_document) == expected

    def test_limit_elements(self, dummy_document):
        prompt = ElementListPrompt(system="sys", user="usr: {elements}", num_elements=3)
        expected = RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="sys"),
                RenderedMessage(
                    role="user",
                    content="usr: ELEMENT 0: Element 1\nELEMENT 1: Element 2\nELEMENT 2: Element 3",
                ),
            ]
        )
        assert prompt.render_document(dummy_document) == expected

    def test_select_odd_elements(self, dummy_document):
        prompt = ElementListPrompt(
            system="sys",
            user="usr: {elements}",
            element_select=lambda elts: [elts[i] for i in range(len(elts)) if i % 2 == 1],
        )
        expected = RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="sys"),
                RenderedMessage(
                    role="user",
                    content="usr: ELEMENT 0: Element 2\nELEMENT 1: Element 4\nELEMENT 2: Element 6",
                ),
            ]
        )
        assert prompt.render_document(dummy_document) == expected

    def test_order_elements(self, dummy_document):
        prompt = ElementListPrompt(system="sys", user="usr: {elements}", element_select=lambda e: list(reversed(e)))
        expected = RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="sys"),
                RenderedMessage(
                    role="user",
                    content="usr: ELEMENT 0: Element 6\nELEMENT 1: Element 5\n"
                    "ELEMENT 2: Element 4\nELEMENT 3: Element 3\nELEMENT 4: Element 2\nELEMENT 5: Element 1",
                ),
            ]
        )
        assert prompt.render_document(dummy_document) == expected

    def test_construct_element_list(self, dummy_document):
        def list_constructor(elts: list[Element]) -> str:
            return "<>" + "</><>".join(f"{i}-{e.type}" for i, e in enumerate(elts)) + "</>"

        prompt = ElementListPrompt(system="sys", user="usr: {elements}", element_list_constructor=list_constructor)
        expected = RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content="sys"),
                RenderedMessage(
                    role="user",
                    content="usr: <>0-Text</><>1-Text</><>2-Text</><>3-Text</><>4-Text</><>5-Text</>",
                ),
            ]
        )
        assert prompt.render_document(dummy_document) == expected

    def test_flattened_properties(self, dummy_document):
        doc = dummy_document.copy()
        doc.properties["entity"] = {"key": "value"}

        prompt = ElementListPrompt(system="sys {doc_property_entity_key}")
        expected = RenderedPrompt(messages=[RenderedMessage(role="system", content="sys value")])
        assert prompt.render_document(doc) == expected
