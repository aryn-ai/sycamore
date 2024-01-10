from abc import ABC, abstractmethod
from typing import Callable, Any, Optional

from ray.data import Dataset

from sycamore.data import Document
from sycamore.plan_nodes import Node, NonCPUUser, NonGPUUser, Transform
from sycamore.utils.generate_ray_func import generate_map_function


class TextAugmentor(ABC):
    @abstractmethod
    def augment_text(self, doc: Document) -> Optional[str]:
        pass

    def __call__(self, doc: Document) -> Optional[str]:
        return self.augment_text(doc)


class UDFTextAugmentor(TextAugmentor):
    """
    UDFTextAugmentor augments text by calling a user-defined function (UDF)
    that maps documents to strings.

    Args:
        fn (Callable[[Document], str]): A function that maps a document to the
            string to use as the new `text_representation`

    Example:
         .. code-block:: python

            def aug_text_fn(doc: Document) -> str:
                return " ".join([
                    f"This pertains to the part {doc.properties['part_name']}.",
                    f"{doc.text_representation}"
                ])
            augmentor = UDFTextAugmentor(aug_text_fn)
            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .augment_text(augmentor)
    """

    def __init__(self, fn: Callable[[Document], str]):
        super().__init__()
        self._fn = fn

    def augment_text(self, doc: Document) -> Optional[str]:
        return self._fn(doc)


class JinjaTextAugmentor(TextAugmentor):
    """
    JinjaTextAugmentor uses a jinja template in a SandboxedEnvironment to
    transform the text representation with metadata from the thingy

    Args:
        template (str): A jinja2 template for the new text represenation. Can contain
            references to `doc` and to any modules passed in the `modules` param
        modules (dict[str, Any]): A mapping of module names to module objects

    Example:
         .. code-block:: python

            from sycamore.transforms.augment_text import JinjaTextAugmentor
            from sycamore.transforms.regex_replace import COALESCE_WHITESPACE
            import pathlib
            template = '''This document is from {{ pathlib.Path(doc.properties['path']).name }}.
            The title is {{ doc.properties['title'] }}.
            The authors are {{ doc.properties['authors'] }}.
            {% if doc.text_representation %}
                {{ doc.text_representation }}
            {% else %}
                There is no text representation for this
            {% endif %}
            '''
            aug = JinjaTextAugmentor(template=template, modules={"pathlib": pathlib})
            aug_docset = exp_docset.augment_text(aug).regex_replace(COALESCE_WHITESPACE)
            aug_docset.show(show_binary=False, truncate_content=False)
    """

    def __init__(self, template: str, modules: dict[str, Any] = {}):
        from jinja2.sandbox import SandboxedEnvironment

        super().__init__()
        self._env = SandboxedEnvironment()
        self._modules = modules
        self._template = template

    def augment_text(self, doc: Document) -> Optional[str]:
        return self._env.from_string(source=self._template, globals=self._modules).render(doc=doc)


class AugmentText(NonCPUUser, NonGPUUser, Transform):
    """
    The AugmentText transform puts metadata into the text representation of
    documents for better embedding and search quality
    """

    def __init__(self, child: Node, text_augmentor: TextAugmentor, **kwargs):
        super().__init__(child, **kwargs)
        self._augmentor = text_augmentor

    def execute(self) -> Dataset:
        input_ds = self.child().execute()

        def augment_text(doc: Document) -> Document:
            doc.text_representation = self._augmentor.augment_text(doc)
            return doc

        output_ds = input_ds.map(generate_map_function(augment_text))
        return output_ds
