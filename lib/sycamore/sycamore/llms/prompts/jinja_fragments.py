J_ELEMENT_LIST = """\
{% for elt in doc.elements[:num_elements] %}ELEMENT {{ elt.element_index }}: {{ elt.field_to_value(field) }}
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
