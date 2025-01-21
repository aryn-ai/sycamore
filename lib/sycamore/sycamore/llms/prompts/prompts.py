from dataclasses import dataclass
from typing import Any, Union, Optional, Callable
import copy

import pydantic
from PIL import Image
from sycamore.data.document import Document, Element


@dataclass
class RenderedMessage:
    """Represents a message per the LLM messages interface - i.e. a role and a content string

    Args:
        role: the role of this message. Should be one of "user", "system", "assistant"
        content: the content of this message, either a python string or a PIL image.
        images: optional list of images to include in this message.
    """

    role: str
    content: str
    images: Optional[list[Image.Image]] = None


@dataclass
class RenderedPrompt:
    """Represents a prompt to be sent to the LLM per the LLM messages interface

    Args:
        messages: the list of messages to be sent to the LLM
        response_format: optional output schema, speicified as pydict/json or
            a pydantic model. Can only be used (iirc) with modern OpenAI models.
    """

    messages: list[RenderedMessage]
    response_format: Union[None, dict[str, Any], pydantic.BaseModel] = None


class SycamorePrompt:
    """Base class/API for all Sycamore LLM Prompt objects. Sycamore Prompts
    convert sycamore objects (``Document``s, ``Element``s) into ``RenderedPrompts``
    """

    def render_document(self, doc: Document) -> RenderedPrompt:
        """Render this prompt, given this document as context.
        Used in llm_map

        Args:
            doc: The document to use to populate the prompt

        Returns:
            A fully rendered prompt that can be sent to an LLM for inference
        """
        raise NotImplementedError(f"render_document is not implemented for {self.__class__.__name__}")

    def render_element(self, elt: Element, doc: Document) -> RenderedPrompt:
        """Render this prompt, given this element and its parent document as context.
        Used in llm_map_elements

        Args:
            elt: The element to use to populate the prompt

        Returns:
            A fully rendered prompt that can be sent to an LLM for inference
        """
        raise NotImplementedError(f"render_element is not implemented for {self.__class__.__name__}")

    def render_multiple_documents(self, docs: list[Document]) -> RenderedPrompt:
        """Render this prompt, given a list of documents as context.
        Used in llm_reduce

        Args:
            docs: The list of documents to use to populate the prompt

        Returns:
            A fully rendered prompt that can be sent to an LLM for inference"""
        raise NotImplementedError(f"render_multiple_documents is not implemented for {self.__class__.__name__}")

    def instead(self, **kwargs) -> "SycamorePrompt":
        """Create a new prompt with some fields changed.

        Args:

            **kwargs: any keyword arguments will get set as fields in the
                resulting prompt

        Returns:

            A new SycamorePrompt with updated fields.

        Example:
             .. code-block:: python

                p = StaticPrompt(system="hello", user="world")
                    p.render_document(Document())
                # [
                #     {"role": "system", "content": "hello"},
                #     {"role": "user", "content": "world"}
                # ]
                p2 = p.instead(user="bob")
                p2.render_document(Document())
                # [
                #     {"role": "system", "content": "hello"},
                #     {"role": "user", "content": "bob"}
                # ]
        """
        new = copy.deepcopy(self)
        for k, v in kwargs.items():
            if hasattr(new, "kwargs") and k not in new.__dict__:
                getattr(new, "kwargs")[k] = v
            else:
                new.__dict__[k] = v
        return new


class ElementListPrompt(SycamorePrompt):
    """A prompt with utilities for constructing a list of elements to include
    in the rendered prompt.

    Args:

        system: The system prompt string. Use {} to reference names that should
            be interpolated. Defaults to None
        user: The user prompt string. Use {} to reference names that should be
            interpolated. Defaults to None
        element_select: Function to choose which set of elements to include in
            the prompt. If None, defaults to the first ``num_elements`` elements.
        element_order: Function to reorder the selected elements. Defaults to
            a noop.
        element_list_constructor: Function to turn a list of elements into a
            string that can be accessed with the interpolation key "{elements}".
            Defaults to "ELEMENT 0: {elts[0].text_representation}\\n
                         ELEMENT 1: {elts[1].text_representation}\\n
                         ..."
        num_elements: Sets the number of elements to take if ``element_select`` is
            unset. Default is 35.
        **kwargs: other keyword arguments are stored and can be used as interpolation keys.

    Example:
         .. code-block:: python

            prompt = ElementListPrompt(
                system = "Hello {name}. This is a prompt about {doc_property_path}"
                user = "What do you make of these tables?\\nTables:\\n{elements}"
                element_select = lambda elts: [e for e in elts if e.type == "table"]
                element_order = reversed
                name = "David Rothschild"
            )
            prompt.render_document(doc)
            # [
            #   {"role": "system", "content": "Hello David Rothschild. This is a prompt about data/mypdf.pdf"},
            #   {"role": "user", "content": "What do you make of these tables?\\nTables:\\n
            #               ELEMENT 0: <last table csv>\\nELEMENT 1: <second-last table csv>..."}
            # ]
    """

    def __init__(
        self,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        element_select: Optional[Callable[[list[Element]], list[Element]]] = None,
        element_order: Optional[Callable[[list[Element]], list[Element]]] = None,
        element_list_constructor: Optional[Callable[[list[Element]], str]] = None,
        num_elements: int = 35,
        **kwargs,
    ):
        super().__init__()
        self.system = system
        self.user = user
        self.element_select = element_select or (lambda elts: elts[:num_elements])
        self.element_order = element_order or (lambda elts: elts)
        self.element_list_constructor = element_list_constructor or (
            lambda elts: "\n".join(f"ELEMENT {i}: {elts[i].text_representation}" for i in range(len(elts)))
        )
        self.kwargs = kwargs

    def _render_element_list_to_string(self, doc: Document):
        elts = self.element_select(doc.elements)
        elts = self.element_order(elts)
        return self.element_list_constructor(elts)

    def render_document(self, doc: Document) -> RenderedPrompt:
        """Render this prompt, given this document as context, using python's
        ``str.format()`` method. The keys passed into ``format()`` are as follows:

            - self.kwargs: the additional kwargs specified when creating this prompt.
            - doc_text: doc.text_representation
            - doc_property_<property_name>: each property name in doc.properties is
                prefixed with 'doc_property_'. So if ``doc.properties = {'k1': 0, 'k2': 3}``,
                you get ``doc_property_k1 = 0, doc_property_k2 = 3``.
            - elements: the element list constructed from doc.elements using ``self.element_select``,
                ``self.element_order``, and ``self.element_list_constructor``.

        Args:
            doc: The document to use as context for rendering this prompt

        Returns:
            A two-message RenderedPrompt containing ``self.system.format()`` and ``self.user.format()``
            using the format keys as specified above.
        """
        format_args = self.kwargs
        format_args["doc_text"] = doc.text_representation
        format_args.update({"doc_property_" + k: v for k, v in doc.properties.items()})
        format_args["elements"] = self._render_element_list_to_string(doc)

        result = RenderedPrompt(messages=[])
        if self.system is not None:
            result.messages.append(RenderedMessage(role="system", content=self.system.format(**format_args)))
        if self.user is not None:
            result.messages.append(RenderedMessage(role="user", content=self.user.format(**format_args)))
        return result


class ElementPrompt(SycamorePrompt):
    """A prompt for rendering an element with utilities for capturing information
    from the element's parent document, with a system and user prompt.

    Args:
        system: The system prompt string. Use {} to reference names to be interpolated.
            Defaults to None
        user: The user prompt string. Use {} to reference names to be interpolated.
            Defaults to None
        include_element_image: Whether to include an image of the element in the rendered user
            message. Only works if the parent document is a PDF. Defaults to False (no image)
        capture_parent_context: Function to gather context from the element's parent document.
            Should return {"key": value} dictionary, which will be made available as interpolation
            keys. Defaults to returning {}
        **kwargs: other keyword arguments are stored and can be used as interpolation keys

    Example:
         .. code-block:: python

            prompt = ElementPrompt(
                system = "You know everything there is to know about {custom_kwarg}, {name}",
                user = "Summarize the information on page {elt_property_page}. \\nTEXT: {elt_text}",
                capture_parent_context = lambda doc, elt: {"custom_kwarg": doc.properties["path"]},
                name = "Frank Sinatra",
            )
            prompt.render_element(doc.elements[0], doc)
            # [
            #   {"role": "system", "content": "You know everything there is to know
            #          about /path/to/doc.pdf, Frank Sinatra"},
            #   {"role": "user", "content": "Summarize the information on page 1. \\nTEXT: <element text>"}
            # ]
    """

    def __init__(
        self,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        include_element_image: bool = False,
        capture_parent_context: Optional[Callable[[Document, Element], dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__()
        self.system = system
        self.user = user
        self.include_element_image = include_element_image
        self.capture_parent_context = capture_parent_context or (lambda doc, elt: {})
        self.kwargs = kwargs

    def render_element(self, elt: Element, doc: Document) -> RenderedPrompt:
        """Render this prompt for this element; also take the parent document
        if there is context in that to account for as well. Rendering is done
        using pythons ``str.format()`` method. The keys passed into ``format``
        are as follows:

            - self.kwargs: the additional kwargs specified when creating this prompt.
            - self.capture_parent_content(doc, elt): key-value pairs returned by the
                context-capturing function.
            - elt_text: elt.text_representation (the text representation of the element)
            - elt_property_<property name>: each property name in elt.properties is
                prefixed with 'elt_property_'. So if ``elt.properties = {'k1': 0, 'k2': 3}``,
                you get ``elt_property_k1 = 0, elt_property_k2 = 3``.

        Args:
            elt: The element used as context for rendering this prompt.
            doc: The element's parent document; used to add additional context.

        Returns:
            A two-message rendered prompt containing ``self.system.format()`` and
            ``self.user.format()`` using the format keys as specified above.
            If self.include_element_image is true, crop out the image from the page
            of the PDF it's on and attach it to the last message (user message if there
            is one, o/w system message).
        """
        format_args = self.kwargs
        format_args.update(self.capture_parent_context(doc, elt))
        format_args["elt_text"] = elt.text_representation
        format_args.update({"elt_property_" + k: v for k, v in elt.properties.items()})

        result = RenderedPrompt(messages=[])
        if self.system is not None:
            result.messages.append(RenderedMessage(role="system", content=self.system.format(**format_args)))
        if self.user is not None:
            result.messages.append(RenderedMessage(role="user", content=self.user.format(**format_args)))
        if self.include_element_image and len(result.messages) > 0:
            from sycamore.utils.pdf_utils import get_element_image

            result.messages[-1].images = [get_element_image(elt, doc)]
        return result


class StaticPrompt(SycamorePrompt):
    """A prompt that always renders the same regardless of the Document or Elements
    passed in as context.

    Args:

        system: the system prompt string. Use {} to reference names to be interpolated.
            Interpolated names only come from kwargs.
        user: the user prompt string. Use {} to reference names to be interpolated.
            Interpolated names only come from kwargs.
        **kwargs: keyword arguments to interpolate.

    Example:
         .. code-block:: python

            prompt = StaticPrompt(system="static", user = "prompt - {number}", number=7)
            prompt.render_document(Document())
            # [
            #   { "role": "system", "content": "static" },
            #   { "role": "user", "content": "prompt - 7" },
            # ]
    """

    def __init__(self, *, system: Optional[str] = None, user: Optional[str] = None, **kwargs):
        super().__init__()
        self.system = system
        self.user = user
        self.kwargs = kwargs

    def render_generic(self) -> RenderedPrompt:
        result = RenderedPrompt(messages=[])
        if self.system is not None:
            result.messages.append(RenderedMessage(role="system", content=self.system.format(**self.kwargs)))
        if self.user is not None:
            result.messages.append(RenderedMessage(role="user", content=self.user.format(**self.kwargs)))
        return result

    def render_element(self, elt: Element, doc: Document) -> RenderedPrompt:
        return self.render_generic()

    def render_document(self, doc: Document) -> RenderedPrompt:
        return self.render_generic()

    def render_multiple_documents(self, docs: list[Document]) -> RenderedPrompt:
        return self.render_generic()
