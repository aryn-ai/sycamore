

import pathlib
from sycamore.data import Document
from sycamore.transforms.augment_text import FStringTextAugmentor, UDFTextAugmentor, JinjaTextAugmentor


class TestAugmentText:
    
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "text",
            "properties": {"path": "/docs/foo.txt", "title": "bar"},
        }
    )

    def test_fstring_augmentation(self):
        aug = FStringTextAugmentor(sentences=[
            "path: {pathlib.Path(doc.properties['path']).name}.",
            "title: {doc.properties['title']}.",
            "exotherm: {doc.properties['exotherm']}.",
            "pure text.",
            "{doc.text_representation}"
        ], modules = [pathlib])
        text = aug.augment_text(self.doc)
        assert text == "path: foo.txt. title: bar. pure text. text"


    def test_udf_augmentation(self, mocker):
        def f(doc: Document) -> str:
            if doc.doc_id == "doc_id":
                return "doc_id"
            else:
                return "not doc id"
        aug = UDFTextAugmentor(f)
        text = aug.augment_text(self.doc)
        assert text == "doc_id"
        text2 = aug.augment_text(Document())
        assert text2 == "not doc id"

    def test_jinja_augmentation(self, mocker):
        template = """{% if doc.properties['path'] %}path: {{ pathlib.Path(doc.properties['path']).name }}.{% endif %}
{% if doc.properties['title'] %}Title: {{ doc.properties['title'] }}.{% endif %}
{% if doc.text_representation %}{{ doc.text_representation }}{% endif %}"""
        aug = JinjaTextAugmentor(template=template, modules={"pathlib": pathlib})
        text = aug.augment_text(self.doc)
        assert text == "path: foo.txt.\nTitle: bar.\ntext"
        text2 = aug.augment_text(Document())
        assert text2 == "\n\n"