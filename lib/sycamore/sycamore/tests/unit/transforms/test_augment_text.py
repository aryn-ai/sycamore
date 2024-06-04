import pathlib
import textwrap
from sycamore.data import Document
from sycamore.transforms.augment_text import UDFTextAugmentor, JinjaTextAugmentor


class TestAugmentText:
    doc = Document(
        {
            "doc_id": "doc_id",
            "type": "pdf",
            "text_representation": "text",
            "properties": {"path": "/docs/foo.txt", "title": "bar"},
        }
    )

    def test_udf_augmentation(self):
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

    def test_jinja_augmentation(self):
        template = textwrap.dedent(
            """\
                    {% if doc.properties['path'] %}path: {{ pathlib.Path(doc.properties['path']).name }}.{% endif %}
                    {% if doc.properties['title'] %}Title: {{ doc.properties['title'] }}.{% endif %}
                    {% if doc.text_representation %}{{ doc.text_representation }}{% endif %}"""
        )
        aug = JinjaTextAugmentor(template=template, modules={"pathlib": pathlib})
        text = aug.augment_text(self.doc)
        print(text)
        assert text == "path: foo.txt.\nTitle: bar.\ntext"
        text2 = aug.augment_text(Document())
        assert text2 == "\n\n"
