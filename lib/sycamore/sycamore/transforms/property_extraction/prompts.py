from typing import Optional
import copy
from sycamore.data import Element, Document
from sycamore.llms.prompts.prompts import (
    ResponseFormat,
    SycamorePrompt,
    RenderedMessage,
    RenderedPrompt,
    compile_templates,
)
from sycamore.llms.prompts.jinja_fragments import J_FORMAT_SCHEMA_MACRO
from sycamore.schema import Schema


class ExtractionJinjaPrompt(SycamorePrompt):
    def __init__(
        self,
        schema: Optional[Schema] = None,
        system: Optional[str] = None,
        user_pre_elements: Optional[str] = None,
        element_template: Optional[str] = None,
        user_post_elements: Optional[str] = None,
        response_format: Optional[ResponseFormat] = None,
        **kwargs,
    ):
        from jinja2.sandbox import SandboxedEnvironment
        from jinja2 import Template

        self.schema = schema
        self.system = system
        self.user_pre_elements = user_pre_elements
        self.user_post_elements = user_post_elements
        self.element_template = element_template
        self.response_format = response_format
        self.kwargs = kwargs
        self._env = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
        self._sys_template: Optional[Template] = None
        self._user_template_1: Optional[Template] = None
        self._elt_template: Optional[Template] = None
        self._user_template_2: Optional[Template] = None

    def __reduce__(self):
        def make(
            schema, system, user_pre_elements, element_template, user_post_elements, response_format, kwargs
        ) -> "ExtractionJinjaPrompt":
            return ExtractionJinjaPrompt(
                schema, system, user_pre_elements, element_template, user_post_elements, response_format, **kwargs
            )

        return make, (
            self.schema,
            self.system,
            self.user_pre_elements,
            self.element_template,
            self.user_post_elements,
            self.response_format,
            self.kwargs,
        )

    def render_multiple_elements(self, elts: list[Element], doc: Document) -> RenderedPrompt:
        if self._elt_template is None:
            templates = compile_templates(
                [self.system, self.user_pre_elements, self.element_template, self.user_post_elements], self._env
            )
            self._sys_template = templates[0]
            self._user_template_1 = templates[1]
            self._elt_template = templates[2]
            self._user_template_2 = templates[3]

        assert self._elt_template is not None, "Unreachable, type narrowing"
        render_args = copy.deepcopy(self.kwargs)
        render_args["schema"] = self.schema
        render_args["doc"] = doc

        messages = []
        if self._sys_template is not None:
            messages.append(RenderedMessage(role="system", content=self._sys_template.render(render_args)))
        if self._user_template_1 is not None:
            messages.append(RenderedMessage(role="user", content=self._user_template_1.render(render_args)))
        for elt in elts:
            messages.append(RenderedMessage(role="user", content=self._elt_template.render(render_args | {"elt": elt})))
        if self._user_template_2 is not None:
            messages.append(RenderedMessage(role="user", content=self._user_template_2.render(render_args)))

        return RenderedPrompt(messages=messages, response_format=self.response_format)


_elt_at_a_time_full_schema = ExtractionJinjaPrompt(
    system="You are a helpful metadata extraction agent. You output only JSON",
    user_pre_elements="""You are provided an element of a document and a schema. Extract all the fields in the
schema as JSON. If a field is not present in the element, output `null` in the output result.""",
    element_template="Element: {{ elt.text_representation }}",
    user_post_elements=J_FORMAT_SCHEMA_MACRO + "Schema: {{ format_schema(schema) }}",
)
