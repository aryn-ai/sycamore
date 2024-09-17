import json
from typing import Optional, Union

from sycamore.data import Document
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
from sycamore.transforms.llm_query import LLMTextQueryAgent
from sycamore.llms import LLM
from sycamore.llms.prompts import ExtractTablePropertiesPrompt, ExtractTablePropertiesTablePrompt


class ExtractTableProperties(SingleThreadUser, NonGPUUser, Map):
    """
    The ExtractKeyValuePair transform extracts the key value from tables and add it as properties to it,
    it only deals with table one level deep.

    Args:
        child: The source node or component that provides the hierarchical documents to be exploded.
        resource_args: Additional resource-related arguments that can be passed to the explosion operation.

    Example:
        .. code-block:: python

            source_node = ...  # Define a source node or component that provides hierarchical documents.
            llm = openAI('gpt-4o-mini')
            property_extract = ExtractKeyValuePair(child=source_node, list=["property_name",llm])
            property_dataset = property_extract.execute()
    """

    def __init__(self, child: Node, parameters: list[Union[str, LLM]], **resource_args):
        super().__init__(child, f=ExtractTableProperties.extract_table_properties, args=parameters, **resource_args)

    @staticmethod
    def extract_parent_json(input_string: str) -> str:
        """
        Extracts the top level JSONstring from input String.
        """
        stack: list[str] = []
        json_start = None

        for i, char in enumerate(input_string):
            if char == "{":
                if not stack:
                    json_start = i
                stack.append(char)
            elif char == "}":
                stack.pop()
                if not stack:
                    json_end = i + 1
                    json_str = input_string[json_start:json_end]
        return json_str

    @staticmethod
    @timetrace("ExtrKeyVal")
    def extract_table_properties(
        parent: Document,
        property_name: str,
        llm: LLM,
        prompt_find_table: Optional[str] = None,
        prompt_LLM: Optional[str] = None,
    ) -> Document:
        """
        This method is used to extract key/value pairs from tables, using the LLM,
        and populate them as a property of that element.
        """
        prompt_find_table = prompt_find_table or ExtractTablePropertiesTablePrompt().user
        query_agent = LLMTextQueryAgent(
            prompt=prompt_find_table, llm=llm, output_property="keyValueTable", element_type="table"
        )
        doc = query_agent.execute_query(parent)

        prompt_llm = prompt_LLM or ExtractTablePropertiesPrompt().user
        query_agent = LLMTextQueryAgent(prompt=prompt_llm, llm=llm, output_property=property_name, element_type="table")
        doc = query_agent.execute_query(parent)

        for ele in doc.elements:
            if ele.type == "table" and property_name in ele.properties.keys():
                if ele.properties.get("keyValueTable", False) != "True":
                    del ele.properties[property_name]
                    continue
                jsonstring_llm = ele.properties.get(property_name)
                assert isinstance(jsonstring_llm, str)
                json_string = ExtractTableProperties.extract_parent_json(jsonstring_llm)
                assert isinstance(json_string, str)
                keyValue = json.loads(json_string)
                if isinstance(keyValue, dict):
                    ele.properties[property_name] = keyValue
                else:
                    raise ValueError(f"Extracted JSON string is not a dictionary: {keyValue}")
        return doc
