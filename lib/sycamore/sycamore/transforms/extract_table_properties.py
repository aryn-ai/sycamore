import json
from typing import Optional, Union

from sycamore.data import Document
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
from sycamore.llms import LLM
from sycamore.llms.prompts import ExtractTablePropertiesPrompt
from PIL import Image
from sycamore.functions.document import split_and_convert_to_image


class ExtractTableProperties(SingleThreadUser, NonGPUUser, Map):
    """
    The ExtractTableProperties transform extracts key-value pairs from tables and adds them as
    properties to the table. It only processes tables that are one level deep.

    Args:
        child: The source node or component that provides the hierarchical documents for extracting table property.
        resource_args: Additional resource-related arguments that can be passed to the extract operation.

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
        json_str = ""

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
        doc1 = split_and_convert_to_image(parent)
        img_list = []
        for img in doc1:
            # print(img['properties'])
            size = tuple(img.properties["size"])
            mode = img.properties["mode"]
            image = Image.frombytes(mode=mode, size=size, data=img.binary_representation)
            img_list.append((image, size, mode))

        for idx, ele in enumerate(parent.elements):
            if ele.type == "table":
                image, size, mode = img_list[ele.properties["page_number"] - 1]  # output of APS is one indexed
                bbox = ele.bbox.coordinates
                img = image.crop((bbox[0] * size[0], bbox[1] * size[1], bbox[2] * size[0], bbox[3] * size[1]))
                content = [
                    {"type": "text", "text": prompt_LLM if prompt_LLM is not None else ExtractTablePropertiesPrompt.user},
                    llm.format_image(img),
                ]
                messages = [
                    {"role": "user", "content": content},
                ]
                prompt_kwargs = {"messages": messages}
                raw_answer = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})
                parsed_json = ExtractTableProperties.extract_parent_json(raw_answer)
                if parsed_json:
                    ele.properties[property_name] = json.loads(parsed_json)
        return parent
