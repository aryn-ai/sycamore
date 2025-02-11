J_ELEMENT_LIST_CAPPED = """\
{% for elt in doc.elements[:num_elements] %}ELEMENT {{ elt.element_index }}: {{ elt.field_to_value(field) }}
{% endfor %}"""

J_ELEMENT_LIST_UNCAPPED = """\
{% for elt in doc.elements %}ELEMENT {{ elt.element_index }}: {{ elt.field_to_value(field) }}
{% endfor %}"""


# Directive to not render the template if the iteration var has
# surpassed the number of batches.
J_BATCH_OOB_CHECK = (
    "{%- if doc.properties[iteration_var] >= doc.properties[batch_key]|count -%}{{ norender() }}{%- endif -%}\n"
)

J_ELEMENT_BATCHED_LIST = (
    J_BATCH_OOB_CHECK
    + """\
{% for i in doc.properties[batch_key][doc.properties[iteration_var]] -%}
{%- set elt = doc.elements[i] -%}
ELEMENT {{ loop.index }}: {{ elt.field_to_value(field) }}
{% endfor -%}"""
)

J_ELEMENT_BATCHED_LIST_WITH_METADATA = (
    J_BATCH_OOB_CHECK
    + """\
{% for i in doc.properties[batch_key][doc.properties[iteration_var]] -%}
{%- set elt = doc.elements[i] -%}
{% if "type" in elt %}Element type: {{ elt.type }}{% endif %}
{% if "page_number" in elt.properties %}Page_number: {{ elt.properties["page_number"] }}{% endif %}
{% if "_element_index" in elt.properties %}Element_index: {{ elt.properties["_element_index"] }}{% endif %}
Text: {{ elt.field_to_value(field) }}
{% endfor -%}"""
)

J_SET_SCHEMA = """{%- if schema is not defined %}{% set schema = doc.properties["_schema"] %}{% endif -%}\n"""
J_SET_ENTITY = (
    """{%- if entity is not defined %}{% set entity = doc.properties.get("_schema_class", "entity") %}{% endif -%}\n"""
)

J_DYNAMIC_DOC_TEXT = (
    """{%- set field = "text_representation" -%}
{% if doc.text_representation is not none %}{{ doc.text_representation }}
{% elif prompt_formatter is defined %}{{ prompt_formatter(doc.elements) }}
{% elif num_elements is defined %}"""
    + J_ELEMENT_LIST_CAPPED
    + "{% else %}"
    + J_ELEMENT_LIST_UNCAPPED
    + "{% endif %}"
)

J_FORMAT_SCHEMA_MACRO = """{%- macro format_schema(schema) -%}
{% for field in schema.fields %}
{{ loop.index }} {{ field.name }}: {{ field.field_type }}: default={{ field.default }}
{% if field.description %}    Decription: {{ field.description }}{% endif %}
{% if field.examples %}    Example values: {{ field.examples }}{% endif %}
{%- endfor -%}
{%- endmacro %}
"""

J_FIELD_VALUE_MACRO = """{%- macro field_value(doc, field, no_field_behavior='none') -%}
{%- if no_field_behavior == "none" -%}{{ doc.field_to_value(field) }}
{%- elif no_field_behavior == "crash" -%}{%- set v = doc.field_to_value(field) -%}
{{ v if v is not none else raise("Could not find field " + field) }}
{%- elif no_field_behavior == "empty" -%}{% set v = doc.field_to_value(field) %}{{ v if v is not none else norender() }}
{%- endif -%}
{%- endmacro -%}
"""
