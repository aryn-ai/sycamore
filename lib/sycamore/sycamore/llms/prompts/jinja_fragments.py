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

J_GET_ELEMENT_TEXT_MACRO = """
{#-
    get_text macro: returns text for an element. If this is the first
    round of summarization:
        If `fields` is provided to the template, add a list of key-value
        pairs to the text (if fields is the string "*", use all properties).
        Always include the text representation
    If this is after the first round of summarization:
        use only the element's summary field
-#}
{%- macro get_text(element, itvarname) %}
    {%- if elt.properties[itvarname] == 0 -%}
        {%- if fields is defined -%}
            {%- if fields == "*" %}{% for p in element.properties %}{% if p.startswith('_') %}{% continue %}{% endif %}
    {{ p }}: {{ element.properties[p] }}
            {% endfor -%}
            {%- else %}{% for f in fields %}
    {{ f }}: {{ element.field_to_value(f) }}
            {% endfor %}{% endif -%}
        {%- endif -%}
    Text: {{ element.text_representation }}
    {%- else -%}
    Summary: {{ element.properties[intermediate_summary_key] }}
    {% endif -%}
{% endmacro -%}
"""

J_HEIRARCHICAL_EXPONENTIAL_COLLECT = """
{%- set exponent = elt.properties[iteration_var] + 1 -%}
{%- if elt.properties[index_key] % (branching_factor ** exponent) != 0 %}{{ norender() }}{% endif -%}
{%- for i in range(elt.properties[index_key], elt.properties[index_key] + (branching_factor ** exponent), branching_factor ** (exponent - 1)) -%}
    {%- if i >= doc.elements|count %}{% break %}{% endif -%}
{{ i }}: {{ get_text(doc.elements[i], iteration_var) }}
{% endfor %}
"""  # noqa: E501 # (line too long)
