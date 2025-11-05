from typing import Optional
import copy
from enum import Enum
from io import BytesIO
import pdf2image
import textwrap

from sycamore.data import Element, Document
from sycamore.llms.prompts.prompts import (
    ResponseFormat,
    SycamorePrompt,
    RenderedMessage,
    RenderedPrompt,
    compile_templates,
)
from sycamore.schema import (
    ArrayProperty,
    ChoiceProperty,
    DateProperty,
    DateTimeProperty,
    ObjectProperty,
    SchemaV2,
    Property,
    DataType,
)
from sycamore.transforms.property_extraction.types import RichProperty
from sycamore.utils.zip_traverse import ZTLeaf, zip_traverse
from sycamore.utils.pdf_utils import get_element_image, select_pdf_pages
import json


def format_schema_v2(schema: SchemaV2, entity_metadata: Optional[RichProperty] = None) -> str:
    obj = schema.as_object_property()
    em = ZTLeaf(None) if entity_metadata is None else entity_metadata
    lines = ["{"]
    propstack: list[Property] = [obj]
    seen_props = {id(obj)}

    for k, (prop, val), (prop_p, val_p) in zip_traverse(obj, em, order="before", intersect_keys=False):
        if prop is None or id(prop) in seen_props:
            continue
        seen_props.add(id(prop))
        prop_p = prop_p.unwrap()
        while len(propstack) > 0 and propstack[-1] is not prop_p:
            p = propstack.pop()
            if isinstance(p, ArrayProperty):
                lines.append("  " * len(propstack) + "]")
            if isinstance(p, ObjectProperty):
                lines.append("  " * len(propstack) + "}")
        prop = prop.unwrap()
        indentation = "  " * len(propstack)
        lines.append("")
        if prop.description is not None:
            lines.append(indentation + f"// Description: {prop.description}")
        if prop.extraction_instructions is not None:
            lines.append(indentation + f"// Extraction Instructions: {prop.extraction_instructions}")
        if len(prop.validators) > 0:
            lines.append(indentation + "// Constraints:")
            for v in prop.validators:
                lines.append(indentation + f"//   - {v.constraint_string()}")
        if prop.examples is not None and len(prop.examples) > 0:
            lines.append(indentation + "// Examples:")
            for ex in prop.examples:
                lines.append(indentation + f"//   - {json.dumps(ex)}")
        if val is not None and len(val.invalid_guesses) > 0:
            lines.append(indentation + "// Invalid Guesses:")
            for ig in val.invalid_guesses:
                lines.append(indentation + f"//   - {json.dumps(ig)}")

        prop_begin = indentation + (f"{k}: " if prop_p.get_type() is not DataType.ARRAY else "")
        required_marker = "" if prop.required else "?"
        if isinstance(prop, ArrayProperty):
            lines.append(prop_begin + "array [")
            propstack.append(prop)
        elif isinstance(prop, ObjectProperty):
            lines.append(prop_begin + "object {")
            propstack.append(prop)
        elif isinstance(prop, ChoiceProperty):
            lines.append(prop_begin + 'enum { "' + '", "'.join(map(str, prop.choices)) + '" }' + required_marker)
        elif isinstance(prop, (DateProperty, DateTimeProperty)):
            lines.append(prop_begin + f"{prop.type.value}{required_marker} ({prop.format})")
        else:
            lines.append(prop_begin + prop.type.value + required_marker)

    while len(propstack) > 0:
        p = propstack.pop()
        if isinstance(p, ArrayProperty):
            lines.append("  " * len(propstack) + "]")
        if isinstance(p, ObjectProperty):
            lines.append("  " * len(propstack) + "}")

    return "\n".join(lines)


class ImageMode(Enum):
    NONE = 0
    PAGE = 1
    ELEMENT = 2


class ExtractionJinjaPrompt(SycamorePrompt):
    def __init__(
        self,
        schema: Optional[SchemaV2 | str] = None,
        system: Optional[str] = None,
        user_pre_elements: Optional[str] = None,
        element_template: Optional[str] = None,
        user_post_elements: Optional[str] = None,
        response_format: Optional[ResponseFormat] = None,
        image_mode: ImageMode = ImageMode.NONE,
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
        self.image_mode = image_mode
        self.kwargs = kwargs
        self._env = SandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
        self._sys_template: Optional[Template] = None
        self._user_template_1: Optional[Template] = None
        self._elt_template: Optional[Template] = None
        self._user_template_2: Optional[Template] = None

    def __reduce__(self):
        def make(
            schema, system, user_pre_elements, element_template, user_post_elements, response_format, image_mode, kwargs
        ) -> "ExtractionJinjaPrompt":
            return ExtractionJinjaPrompt(
                schema,
                system,
                user_pre_elements,
                element_template,
                user_post_elements,
                response_format,
                image_mode,
                **kwargs,
            )

        return make, (
            self.schema,
            self.system,
            self.user_pre_elements,
            self.element_template,
            self.user_post_elements,
            self.response_format,
            self.image_mode,
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

        # assert self._elt_template is not None, "Unreachable, type narrowing"
        render_args = copy.deepcopy(self.kwargs)
        if self.schema is not None:
            if isinstance(self.schema, SchemaV2):
                render_args["schema"] = format_schema_v2(self.schema)
            else:
                render_args["schema"] = self.schema
        render_args["doc"] = doc

        messages = []
        if self._sys_template is not None:
            messages.append(RenderedMessage(role="system", content=self._sys_template.render(render_args)))
        if self._user_template_1 is not None or self.image_mode == ImageMode.PAGE:
            m = RenderedMessage(role="user", content="")
            if self._user_template_1 is not None:
                m.content = self._user_template_1.render(render_args)
            if self.image_mode == ImageMode.PAGE:
                assert (
                    doc.binary_representation is not None
                ), f"Cannot extract from an image because the document has no binary: {doc}"
                pages = {e.properties.get("page_number", -1) for e in elts}
                pages.discard(-1)
                if len(pages) > 0:
                    bits = BytesIO(doc.binary_representation)
                    pagebits = BytesIO()
                    select_pdf_pages(bits, pagebits, list(pages))
                    images = pdf2image.convert_from_bytes(pagebits.getvalue())
                    m.images = images

            messages.append(m)
        if self._elt_template is not None:
            for elt in elts:
                m = RenderedMessage(role="user", content=self._elt_template.render(render_args | {"elt": elt}))
                if self.image_mode == ImageMode.ELEMENT:
                    m.images = [get_element_image(elt, doc)]
                messages.append(m)
        if self._user_template_2 is not None:
            messages.append(RenderedMessage(role="user", content=self._user_template_2.render(render_args)))

        return RenderedPrompt(messages=messages, response_format=self.response_format)


extract_system = """\
You are a helpful metadata extraction agent. You output only JSON. Make sure the JSON you output is valid.

- Numerical values must contain only numeric characters and up to one decimal point; e.g. 3,201.6 should be returned as 3201.6
- Numerical values MUST NOT contain any non-numeric characters, including '?', '_', ','. Don't mess this up!
- Date/Datetime values should always be quoted, e.g. 2025-09-04 should be returned as "2025-09-04"
- Values must not contain any mathematical expressions. If necessary, preform the calculation yourself.
- Quotes in strings must be properly escaped.
- Always output an object type at the root level, e.g. {"key": "value"}, not a list.

For array fields, extract **every** instance of the type described in the array.
"""

_elt_at_a_time_full_schema = ExtractionJinjaPrompt(
    system=extract_system,
    user_pre_elements="""You are provided some elements of a document and a schema. Extract all the fields in the
schema as JSON. If a field is not present in the element, output `null` in the output result.""",
    element_template="Element: {{ elt.text_representation }}",
    user_post_elements="Schema: \n```\n{{ schema }}\n```",
)

_page_image_full_schema = ExtractionJinjaPrompt(
    system=extract_system,
    user_pre_elements="""You are provided a page of a document and a schema. Extract all the fields in the schema
as JSON. If a field is not present on the page, output `null` in the output result.""",
    user_post_elements="Schema: \n```\n{{ schema }}\n```",
    image_mode=ImageMode.PAGE,
)

default_prompt = _elt_at_a_time_full_schema

schema_extract_pre_elements_helper = """\n
You are given a schema that has already been extracted from the document. Now extract only the new properties that are missing from this schema. Do not include any properties that are already in the schema. Use the structure of the schema (names and nesting) to decide what is already included.
Extracted schema:
{existing_schema}
"""

schema_extraction_system_prompt = textwrap.dedent(
    """\
    You are a helpful property extractor. You only return a JSON schema which is a list of properties as defined below. Return only the relevant properties; for example, if it's a form, you might want to return several properties whereas if it's an article, you might want to return only the relevant properties. Be very careful about what properties you return.
    """
)

_schema_extraction_prompt = ExtractionJinjaPrompt(
    system=schema_extraction_system_prompt,
    user_pre_elements=textwrap.dedent(
        """\
        Extract a JSON schema from the {{ element_description }}.

        Each property must have:
        - `name`: lowercase, underscore-separated string representing the name of the property. The name should be descriptive and concise. It should describe the kind of property, **not its value**.
        - `type`: the parent object of the property which contains the following fields:
            - `type`: one of "bool", "int", "float", "string", "date", "datetime", "array", "object", "choice"
            - `description`: a brief human-readable explanation of what the property represents
            - `examples`: the values of the property extracted from the document.
        - For properties of type `array`, include `item_type` describing the type of items (can be "bool", "int", "float", "string", "date", "datetime", "array", "object", "choice")
        - For properties of type `object`, include `properties`, a list of named sub-properties following the same schema
        - For properties of type `choice`, include `choices`, a list of possible values for the property

        Guidelines:
        - Nested properties must follow this schema recursively
        - Do not return any explanation or extra text outside the JSON
        - Examples MUST be present ONLY at the top level of each property (never in sub-properties). For each property, include at most 5 examples; if there are more, select the most representative 5. NEVER include an 'examples' field in any nested sub-property.
        - If a property has no examples, assign its 'examples' field to null.
        - properties of type "bool" should be either `true` or `false`
        - properties of type "date" should be in ISO format (YYYY-MM-DD)
        - properties of type "datetime" should be in ISO format (YYYY-MM-DDTHH:MM:SS)

        {{ additional_instructions }}

        Example output:
        [
            {
                'name': 'company_name',
                'type':
                    {
                        'type': 'string',
                        'examples': ['Acme Corp', 'Globex Corporation'],
                        'description': 'The name of the company'
                    }
            },
            {
                'name': 'founded_year',
                'type':
                    {
                        'type': 'int',
                        'examples': [1999, 2001],
                        'description': 'The year the company was founded'
                    }
            },
            {
                'name': 'is_public',
                'type':
                    {
                        'type': 'bool',
                        'examples': [true, false],
                        'description': 'Whether the company is publicly listed'
                    }
            },
            {
                'name': 'headquarters',
                'type':
                    {
                        'type': 'object',
                        'properties': [
                            {
                                'name': 'city',
                                'type':
                                    {
                                        'type': 'string',
                                        'description': 'City of the headquarters'
                                    }
                            },
                            {
                                'name': 'state',
                                'type':
                                    {
                                        'type': 'string',
                                        'description': 'State of the headquarters'
                                    }
                            }
                        ],
                        'description': "Information about the headquarters",
                        'examples': [{'city': 'San Francisco', 'state': 'CA'}, {'city': 'New York', 'state': 'NY'}]
                    }
            },
            {
                'name': 'tags',
                'type':
                    {
                        'type': 'array',
                        'item_type': {
                            'type': 'string',
                            'description': 'A tag associated with the company'
                        },
                        'examples': [['technology', 'innovation'], ['finance', 'investment']],
                        'description': 'List of tags associated with the company'
                    }
            },
            {
                'name': 'status',
                'type':
                    {
                        'type': 'choice',
                        'choices': ['active', 'inactive', 'acquired'],
                        'examples': ['active', 'inactive'],
                        'description': 'Current status of the company'
                    }
            }
        ]
        """
    ),
    element_template="Element {{ elt.element_index }}: {{ elt.text_representation }}",
)
