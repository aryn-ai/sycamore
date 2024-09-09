from sycamore.data import Document
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
import json
from typing import Union
import logging
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
        parent: Document, property_name: str, llm: LLM, prompt_find_table: str = "", prompt_LLM: str = ""
    ) -> Document:
        """
        This Method is used to extract key value pair from table using LLM and
        populate it as property of that element.
        """
        if prompt_find_table == "":
            prompt_find_table = ExtractTablePropertiesTablePrompt().user
        query_agent = LLMTextQueryAgent(
            prompt=prompt_find_table, llm=llm, output_property="keyValueTable", element_type="table"
        )
        doc = query_agent.execute_query(parent)

        if prompt_LLM == "":
            prompt_LLM = ExtractTablePropertiesPrompt().user
        query_agent = LLMTextQueryAgent(prompt=prompt_LLM, llm=llm, output_property=property_name, element_type="table")
        doc = query_agent.execute_query(parent)

        for ele in doc.elements:
            if ele.type == "table" and property_name in ele.properties.keys():
                try:
                    if ele.properties.get("keyValueTable", False) != "True":
                        del ele.properties[property_name]
                        continue
                    jsonstring_llm = ele.properties.get(property_name)
                    assert isinstance(jsonstring_llm, str)
                    json_string = ExtractTableProperties.extract_parent_json(jsonstring_llm)
                    assert isinstance(json_string, str)
                    keyValue = json.loads(json_string)
                    ele.properties[property_name] = keyValue
                except Exception as e:
                    logging.error(str(e))
        return doc
