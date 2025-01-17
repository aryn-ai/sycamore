from dataclasses import dataclass
from typing import Any, Union, Optional, Callable
import copy

import pydantic
from sycamore.data.document import Document, Element


@dataclass
class RenderedMessage:
    """Represents a message per the LLM messages interface - i.e. a role and a content string

    Args:
        role: the role of this message. Should be one of "user", "system", "assistant"
        content: the content of this message.
    """

    role: str
    content: str

    def to_dict(self):
        return {"role": self.role, "content": self.content}


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

    def to_dict(self):
        res = {"messages": [m.to_dict() for m in self.messages]}
        if self.response_format is not None:
            res["response_format"] = self.output_structure  # type: ignore
        return res


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

    def render_element(self, elt: Element) -> RenderedPrompt:
        """Render this prompt, given this element as context.
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
        new.__dict__.update(kwargs)
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
            Defaults to "ELEMENT 0: {elts[0].text_representation}\n
                         ELEMENT 1: {elts[1].text_representation}\n
                         ..."
        num_elements: Sets the number of elements to take if ``element_select`` is
            unset. Default is 35.
        **kwargs: other keyword arguments are stored and can be used as interpolation keys.

    Example:
         .. code-block:: python

            prompt = ElementListPrompt(
                system = "Hello {name}. This is a prompt about {doc_property_path}"
                user = "What do you make of these tables?\nTables:\n{elements}"
                element_select = lambda elts: [e for e in elts if e.type == "table"]
                element_order = reversed
                name = "David Rothschild"
            )
            prompt.render_document(doc)
            # [
            #   {"role": "system", "content": "Hello David Rothschild. This is a prompt about data/mypdf.pdf"},
            #   {"role": "user", "content": "What do you make of these tables?\nTables:\n
            #               ELEMENT 0: <last table csv>\nELEMENT 1: <second-last table csv>..."}
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
        self.system = system
        self.user = user
        self.element_select = element_select or (lambda elts: elts[:num_elements])
        self.element_order = element_order or (lambda elts: elts)
        self.element_list_constructor = element_list_constructor or (
            lambda elts: "\n".join(f"ELEMENT {i}: {elts[i].text_representation}" for i in range(len(elts)))
        )
        self.kwargs = kwargs
        super().__init__()

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

    def instead(self, **kwargs) -> "SycamorePrompt":
        new = copy.deepcopy(self)
        for k in kwargs:
            if k in new.__dict__:
                new.__dict__[k] = kwargs[k]
            else:
                new.kwargs[k] = kwargs[k]
        return new
