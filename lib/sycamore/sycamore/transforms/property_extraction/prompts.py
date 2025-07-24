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
import textwrap


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

_schema_extraction_prompt = ExtractionJinjaPrompt(
    system="You are a helpful entity extractor. You only return a JSON list of entities. Return only the relevant entities; for example, if it's a form, you might want to return several entities whereas if it's an article, you might want to return only the relevant entities. Be very careful about what entities you return.",
    user_pre_elements=textwrap.dedent(
        """\
        Extract a flat JSON list of entities from the following document text.

        Each entity must have:
        - `name`: lowercase, underscore-separated string representing the name of the entity. The name should be descriptive and concise. It should describe the kind of entity, **not its value**. 
        - `value`: the value of the entity extracted from the document
        - `type`: one of: "string", "integer", "float", "date", "datetime".
        - `description`: a brief human-readable explanation of what the entity represents

        Guidelines:
        - Use a flat schema (no nested properties)
        - Do not return any explanation or extra text outside the JSON
        - Entity of type "date" should be in ISO format (YYYY-MM-DD)
        - Entity of type "datetime" should be in ISO format (YYYY-MM-DDTHH:MM:SS)

        Example output:
        [
            {"name": "company_name", "value": "Acme Corp", "type": "string", "description": "The name of the company"},
            {"name": "ceo", "value": "Jane Doe", "type": "string", "description": "The CEO of the company"}
        ]
        """
    ),
    element_template="Element {{ elt.element_index }}: {{ elt.text_representation }}",
)
