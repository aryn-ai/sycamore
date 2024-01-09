from abc import ABC, abstractmethod
from typing import Callable, Any

from ray.data import Dataset

from sycamore.data import Document
from sycamore.plan_nodes import Node, NonCPUUser, NonGPUUser, Transform
from sycamore.utils.generate_ray_func import generate_map_class_from_callable, generate_map_function




class TextAugmentor(ABC):
    @abstractmethod
    def augment_text(self, doc: Document) -> str:
        pass

class FStringTextAugmentor(TextAugmentor):
    """
    FStringTextAugmentor augments text by means of a list of python format-strings. 
    The only top-level replacement variable available for use in such strings is `doc`,
    which represents a document object. The inputted format-strings will be joined with
    whitespace separators. Any format-strings that fail due to a missing property will
    be silently dropped, but any format strings that ask for other replacement variables
    will throw errors.

    Note: do not include the "f" prefix for the f-string. i.e. instead of 
        `f"expression: {doc.text_representation}"`
    use
        `"expression: {doc.text_representation}"`

    Args:
        sentences (list[str]): List of sentences that optionally contain `doc` replacement variable

    Example:
         .. code-block:: python

            from sycamore.transforms.augment_text import FStringTextAugmentor
            import pathlib
            augmentor = FStringTextAugmentor([
                "This is from {pathlib.Path(doc.properties['path']).name}.",
                "The title of this paper is {doc.properties['title']}.",
                "The authors are {doc.properties['authors']}.",
                "{doc.text_representation}"
            ], modules=[pathlib])
            aug_docset = exp_docset.augment_text(augmentor=augmentor)
    """

    def __init__(self, sentences: list[str], modules: list = []):
        super().__init__()
        self._sentences = [compile('f"' + s + '"', "<string>", "eval") for s in sentences]
        self._modules = {m.__name__: m for m in modules}

    def augment_text(self, doc: Document) -> str:
        formatted = []
        for s in self._sentences:
            try:
                evaluated = eval(s, self._modules, {"doc": doc})
                formatted.append(evaluated)
            except Exception as e:
                print(f"augment_text: Document {doc.doc_id} failed in sentence {s} with error {e}")

        if len(formatted) == 0:
            print(f"augment_text: Document {doc.doc_id} ended up with empty text representation. Using original text instead")
            return doc.text_representation
        return " ".join(formatted)
    
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

    def augment_text(self, doc: Document) -> str:
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

    def augment_text(self, doc: Document) -> str:
        return self._env.from_string(source=self._template, globals=self._modules).render(doc = doc)

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