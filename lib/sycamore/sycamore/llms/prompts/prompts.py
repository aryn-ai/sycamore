from dataclasses import dataclass
from typing import Any, Union, Optional, Callable, TYPE_CHECKING
from typing_extensions import Self
import copy

import pydantic
from PIL import Image
from sycamore.data.document import Document, Element
from sycamore.functions.tokenizer import Tokenizer
from sycamore.connectors.common import flatten_data

if TYPE_CHECKING:
    from jinja2.sandbox import SandboxedEnvironment
    from jinja2 import Template


ResponseFormat = Union[None, dict[str, Any], type[pydantic.BaseModel]]


@dataclass
class RenderedMessage:
    """Represents a message per the LLM messages interface - i.e. a role and a content string

    Args:
        role: the role of this message. e.g. for OpenAI should be one of "user", "system", "assistant"
        content: the content of this message
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
            a pydantic model. Can only be used with modern OpenAI models.
    """

    messages: list[RenderedMessage]
    response_format: ResponseFormat = None

    def token_count(self, tokenizer: Tokenizer) -> int:
        if len(self.messages) == 0:
            return 0
        return sum(len(tokenizer.tokenize(m.content)) for m in self.messages)


class SycamorePrompt:
    """Base class/API for all Sycamore LLM Prompt objects. Sycamore Prompts
    convert sycamore objects (``Document``, ``Element``) into ``RenderedPrompts``
    """

    def render_any(self, **kwargs) -> RenderedPrompt:
        """Render this prompt, given the input data as context

        Args:
            **kwargs: key-value pairs of data to include in the prompt

        Returns:
            A fully rendered prompt that can be sent to an llm for inference
        """
        raise NotImplementedError(f"render_any is not implemented for {self.__class__.__name__}")

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

    def fork(self, **kwargs: Any) -> Self:
        """Create a new prompt with some fields changed.

        Args:
            ignore_none: bool. do not set any kwargs with value `None`. This is not in the
                method signature because mypy sucks. https://github.com/python/mypy/issues/17642
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
                p2 = p.set(user="bob")
                p2.render_document(Document())
                # [
                #     {"role": "system", "content": "hello"},
                #     {"role": "user", "content": "bob"}
                # ]
        """
        ignore_none = kwargs.pop("ignore_none", False)
        new = copy.deepcopy(self)
        for k, v in kwargs.items():
            if ignore_none and v is None:
                continue
            if hasattr(new, "kwargs") and k not in new.__dict__:
                getattr(new, "kwargs")[k] = v
            else:
                new.__dict__[k] = v
        return new


def _build_format_str(
    system: Optional[str], user: Union[None, str, list[str]], format_args: dict[str, Any]
) -> list[RenderedMessage]:
    messages = []
    if system is not None:
        messages.append(RenderedMessage(role="system", content=system.format(**format_args)))
    if isinstance(user, list):
        messages.extend([RenderedMessage(role="user", content=u.format(**format_args)) for u in user])
    elif isinstance(user, str):
        messages.append(RenderedMessage(role="user", content=user.format(**format_args)))
    return messages


class ElementListPrompt(SycamorePrompt):
    """A prompt with utilities for constructing a list of elements to include
    in the rendered prompt.

    Args:

        system: The system prompt string. Use {} to reference names that should
            be interpolated. Defaults to None
        user: The user prompt string. Use {} to reference names that should be
            interpolated. Defaults to None
        element_select: Function to choose the elements (and their order) to include
            in the prompt. If None, defaults to the first ``num_elements`` elements.
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
                element_select = lambda elts: list(reversed(e for e in elts if e.type == "table"))
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
        user: Union[None, str, list[str]] = None,
        element_select: Optional[Callable[[list[Element]], list[Element]]] = None,
        element_list_constructor: Optional[Callable[[list[Element]], str]] = None,
        num_elements: int = 35,
        **kwargs,
    ):
        super().__init__()
        self.system = system
        self.user = user
        self.element_select = element_select or (lambda elts: elts[:num_elements])
        self.element_list_constructor = element_list_constructor or (
            lambda elts: "\n".join(f"ELEMENT {i}: {elts[i].text_representation}" for i in range(len(elts)))
        )
        self.kwargs = kwargs

    def _render_element_list_to_string(self, doc: Document):
        elts = self.element_select(doc.elements)
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
        format_args = copy.deepcopy(self.kwargs)
        format_args["doc_text"] = doc.text_representation or self._render_element_list_to_string(doc)
        flat_props = flatten_data(doc.properties, prefix="doc_property", separator="_")
        format_args.update(flat_props)
        format_args["elements"] = self._render_element_list_to_string(doc)

        messages = _build_format_str(self.system, self.user, format_args)
        result = RenderedPrompt(messages=messages)
        return result


class ElementListIterPrompt(ElementListPrompt):
    """A prompt with utilities for constructing a lists of elements to include
    in a sequence of rendered prompts. Functions almost identically to ElementListPrompt,
    but renders into a series of prompts.

    Args:

        system: The system prompt string. Use {} to reference names that should
            be interpolated. Defaults to None
        user: The user prompt string. Use {} to reference names that should be
            interpolated. Defaults to None
        element_select: Function to choose the elements (and their order) to include
            in the prompt. If None, defaults to the first ``num_elements`` elements.
        element_list_constructor: Function to turn a list of elements into a
            string that can be accessed with the interpolation key "{elements}".
            Defaults to "ELEMENT 0: {elts[0].text_representation}\\n
                         ELEMENT 1: {elts[1].text_representation}\\n
                         ..."
        num_elements: Sets the number of elements to take if ``element_select`` is
            unset. Default is 35.
        element_batcher: Constructs batches of elements to render in sequence to generate
            several rendered prompts. Defaults to one batch with all elements.
        iteration_var_name: Name of the property to look for in the document to determine
            which batch of elements to use to render the prompt. Default is "i"
        **kwargs: other keyword arguments are stored and can be used as interpolation keys.

    Example:
         .. code-block:: python

            prompt = ElementListIterPrompt(
                system = "You are a program that returns 'None' if you don't know the answer to my question"
                user = "What is the capital of the country described?\\nElements:\\n{elements}"
                element_batcher = lambda elts: [elts[i:i+2] for i in range(0, len(elts), 2)]
            ).set(is_done=lambda s: s != 'None')
            doc.properties["i"] = 0
            prompt.render_document(doc)
            # [
            #     {"role": "system", "content": "You are a program that returns 'None' if you don't
            #             know the answer to my question"},
            #     {"role": "user", "content": "What is the capital of the country described?\\nElements:\\n
            #             ELEMENT 0: <elt 0 text>\\nELEMENT 1: <elt 1 text>"}
            # ]
            doc.properties["i"] = 1
            prompt.render_document(doc)
            # [
            #     {"role": "system", "content": "You are a program that returns 'None' if you don't
            #             know the answer to my question"},
            #     {"role": "user", "content": "What is the capital of the country described?\\nElements:\\n
            #             ELEMENT 0: <elt 2 text>\\nELEMENT 1: <elt 3 text>"}
            # ]
    """

    def __init__(
        self,
        *,
        element_batcher: Optional[Callable[[list[Element]], list[list[Element]]]] = None,
        iteration_var_name: str = "i",
        **kwargs,
    ):
        self.element_batcher = element_batcher or (lambda e: [e])
        self.iteration_var_name = iteration_var_name
        super().__init__(**kwargs)

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
            A two-message RenderedPrompt containing ``self.system.format()`` and
            ``self.user.format()`` using the format keys as specified above. The prompt is
            rendered from the ``doc.properties[self.iteration_var_name]``'th batch of
            elements generated by ``self.element_batcher``
        """
        i = doc.properties.get(self.iteration_var_name, 0)

        format_args = copy.deepcopy(self.kwargs)
        format_args["doc_text"] = doc.text_representation
        flat_props = flatten_data(doc.properties, prefix="doc_property", separator="_")
        format_args.update(flat_props)

        for j, elt_batch in enumerate(self.element_batcher(doc.elements)):
            if j < i:
                continue
            else:
                elements = self.element_select(elt_batch)
                elementstr = self.element_list_constructor(elements)
                messages = _build_format_str(self.system, self.user, {"elements": elementstr, **format_args})
                return RenderedPrompt(messages=messages)
        return RenderedPrompt(messages=[])


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
        user: Union[None, str, list[str]] = None,
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
        format_args = copy.deepcopy(self.kwargs)
        format_args.update(self.capture_parent_context(doc, elt))
        format_args["elt_text"] = elt.text_representation
        flat_props = flatten_data(elt.properties, prefix="elt_property", separator="_")
        format_args.update(flat_props)

        messages = _build_format_str(self.system, self.user, format_args)
        result = RenderedPrompt(messages=messages)
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

    def __init__(self, *, system: Optional[str] = None, user: Union[None, str, list[str]] = None, **kwargs):
        super().__init__()
        self.system = system
        self.user = user
        self.kwargs = kwargs

    def render_generic(self) -> RenderedPrompt:
        messages = _build_format_str(self.system, self.user, self.kwargs)
        result = RenderedPrompt(messages=messages)
        return result

    def render_element(self, elt: Element, doc: Document) -> RenderedPrompt:
        return self.render_generic()

    def render_document(self, doc: Document) -> RenderedPrompt:
        return self.render_generic()

    def render_multiple_documents(self, docs: list[Document]) -> RenderedPrompt:
        return self.render_generic()


class PromptNoRender(Exception):
    def __init__(self):
        super().__init__()


class PromptException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg


def raise_no_render():
    raise PromptNoRender()


def raise_uncaught(msg: str):
    raise PromptException(msg)


def compile_templates(templates: list[Optional[str]], env: "SandboxedEnvironment") -> list[Optional["Template"]]:
    return [
        (
            env.from_string(source=t, globals={"norender": raise_no_render, "raise": raise_uncaught})
            if t is not None
            else None
        )
        for t in templates
    ]


def render_templates(sys: Optional["Template"], user: list["Template"], render_args: dict[str, Any]) -> RenderedPrompt:
    messages = []
    if sys is not None:
        try:
            system = sys.render(render_args)
            messages.append(RenderedMessage(role="system", content=system))
        except PromptNoRender:
            return RenderedPrompt(messages=[])
    for ut in user:
        try:
            content = ut.render(render_args)
            messages.append(RenderedMessage(role="user", content=content))
        except PromptNoRender:
            return RenderedPrompt(messages=[])
    return RenderedPrompt(messages=messages)


def _deserialize_jinja_prompt(kwargs):
    cls = kwargs.pop("class")
    if cls == "JinjaPrompt":
        return JinjaPrompt(**kwargs)
    if cls == "JinjaElementPrompt":
        return JinjaElementPrompt(**kwargs)


class JinjaPrompt(SycamorePrompt):
    """A prompt that uses the Jinja templating system to render documents, with
    a system and user prompt.

    Args:
        system: The system prompt template, using Jinja syntax.
        user: The user prompt template or prompt templates, using Jinja syntax.
        response_format: Optional constraint on the format of the model output
        kwargs: Additional key-value pairs that will be made available to the
            rendering engine.

    Example:
         .. code-block:: python

            prompt = JinjaPrompt(
                system="You are a helpful entity extractor that extracts a json object or list to"
                        " populate a data processing system",
                user='''Below, you will be given a series of segments of an NTSB report and a question.
            Your job is to provide the answer to the question based on the value provided.
            Your response should ONLY contain the answer to the question. If you are not able
            to extract the new field given the information, respond with "None". The type
            of your response should be a JSON list of strings.
            Field value:
            {% for elt in doc.elements[:10] %}
            ELEMENT {{ elt.element_index }}: {{ elt.field_to_value(field) }}
            {% endfor %}
            Answer the question "{{ question }}":''',
                question="What aircraft parts were damaged in this report?",
                field="text_representation",
            )
            ds.llm_map(prompt, output_field="damaged_parts", llm=OpenAI(OpenAIModels.GPT_4O))

    """

    def __init__(
        self,
        *,
        system: Optional[str] = None,
        user: Union[None, str, list[str]] = None,
        response_format: ResponseFormat = None,
        **kwargs,
    ):
        from jinja2.sandbox import SandboxedEnvironment
        from jinja2 import Template

        super().__init__()
        self.system = system
        self.user = user
        self.response_format = response_format
        self.kwargs = kwargs
        self._env = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
        self._sys_template: Optional[Template] = None
        self._user_templates: Union[None, list[Template]] = None

    def __reduce__(self):
        # Cannot serialize compiled templates - so force recompilation
        return _deserialize_jinja_prompt, (
            {
                "system": self.system,
                "user": self.user,
                "class": self.__class__.__name__,
                "response_format": self.response_format,
                **self.kwargs,
            },
        )

    def render_document(self, doc: Document) -> RenderedPrompt:
        """Render this document using Jinja's template rendering system.
        The template gets references to:

            - doc: the document
            - **self.kwargs: other keyword arguments held by this prompt are
                available by name.

        Args:
            doc: The document to render

        Returns:
            A rendered prompt containing information from the document.
        """
        if self._user_templates is None:
            userlist = self.user if isinstance(self.user, list) else [self.user]  # type: ignore
            templates = compile_templates([self.system] + userlist, self._env)  # type: ignore
            self._sys_template = templates[0]
            self._user_templates = [t for t in templates[1:] if t is not None]

        render_args = copy.deepcopy(self.kwargs)
        render_args["doc"] = doc

        rendered = render_templates(self._sys_template, self._user_templates, render_args)
        if self.response_format is not None:
            rendered.response_format = self.response_format
        return rendered


class JinjaElementPrompt(SycamorePrompt):
    """A prompt that uses the Jinja templating system to render elements, with
    a system and user prompt.

    Args:
        system: The system prompt template, using Jinja syntax.
        user: The user prompt template or prompt templates, using Jinja syntax.
        include_image: Whether to include the image of the element in the rendered prompt. Default is False
        response_format: Optional response format constraint for the LLM.
        kwargs: Additional key-value pairs that will be made available to the
            rendering engine.

    Example:
         .. code-block:: python

            prompt = JinjaElementPrompt(
                system="You are a helpful entity extractor that extracts a json object or list to"
                        " populate a data processing system",
                user='''Below, you will be given a segment of an NTSB report and a question.
            Your job is to provide the answer to the question based on the value provided.
            Your response should ONLY contain the answer to the question. If you are not able
            to extract the new field given the information, respond with "None". The type
            of your response should be a JSON list of strings.
            Field value:
            ELEMENT {{ elt.element_index }}: {{ elt.field_to_value(field) }}

            Answer the question "{{ question }}":''',
                question="What aircraft parts were damaged in this report?",
                field="text_representation",
            )
            ds.llm_map(prompt, output_field="damaged_parts", llm=OpenAI(OpenAIModels.GPT_4O))

    """

    def __init__(
        self,
        *,
        system: Optional[str] = None,
        user: Union[None, str, list[str]] = None,
        include_image: bool = False,
        response_format: ResponseFormat = None,
        **kwargs,
    ):
        from jinja2.sandbox import SandboxedEnvironment
        from jinja2 import Template

        super().__init__()
        self.system = system
        self.user = user
        self.include_image = include_image
        self.response_format = response_format
        self.kwargs = kwargs
        self._env = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
        self._sys_template: Optional[Template] = None
        self._user_templates: Union[None, list[Template]] = None

    def __reduce__(self):
        # Cannot serialize compiled templates - so force recompilation
        return _deserialize_jinja_prompt, (
            {
                "system": self.system,
                "user": self.user,
                "include_image": self.include_image,
                "response_format": self.response_format,
                "class": self.__class__.__name__,
                **self.kwargs,
            },
        )

    def render_element(self, elt: Element, doc: Document) -> RenderedPrompt:
        """Render this document using Jinja's template rendering system.
        The template gets references to:

            - elt: the element
            - doc: the document containing the element
            - **self.kwargs: other keyword arguments held by this prompt are
                available by name.

        Args:
            elt: The element to render
            doc: The document containing the element

        Returns:
            A rendered prompt containing information from the element.
        """
        if self._user_templates is None:
            userlist = self.user if isinstance(self.user, list) else [self.user]  # type: ignore
            templates = compile_templates([self.system] + userlist, self._env)  # type: ignore
            self._sys_template = templates[0]
            self._user_templates = [t for t in templates[1:] if t is not None]

        render_args = copy.deepcopy(self.kwargs)
        render_args["elt"] = elt
        render_args["doc"] = doc

        result = render_templates(self._sys_template, self._user_templates, render_args)
        if self.include_image and len(result.messages) > 0:
            from sycamore.utils.pdf_utils import get_element_image

            result.messages[-1].images = [get_element_image(elt, doc)]
        if self.response_format is not None:
            result.response_format = self.response_format
        return result
