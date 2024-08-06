from sycamore.data import Document
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
import json
from typing import List, Dict
import sycamore 
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import ArynPartitioner
import logging
from sycamore.transforms.llm_query import LLMTextQueryAgent
from sycamore.llms import OpenAI
import pickle

class ExtractKeyValuePair(SingleThreadUser, NonGPUUser, Map):
    """
    The ExtractKeyValuePair transform extracts the key value from a table and add it as properties to the table element, 
    it only deals with table one level deep. 

    Args:
        child: The source node or component that provides the hierarchical documents to be exploded.
        resource_args: Additional resource-related arguments that can be passed to the explosion operation.

    Example:
        .. code-block:: python

            source_node = ...  # Define a source node or component that provides hierarchical documents.
            property_transform = AssignDocProperties(child=source_node, list=["table"])
            property_dataset = property_transform.execute()
    """

    def __init__(self, child: Node, parameters: List[str], **resource_args):
        super().__init__(child, f=ExtractKeyValuePair.extract_table_properties, args=parameters, **resource_args)

    @staticmethod
    def extract_parent_json(input_string):
        stack = []
        json_start = None

        for i, char in enumerate(input_string):
            if char == '{':
                if not stack:
                    json_start = i
                stack.append(char)
            elif char == '}':
                stack.pop()
                if not stack:
                    json_end = i + 1
                    json_str = input_string[json_start:json_end]
        return json_str
    
    @timetrace("ExtrKeyVal")
    @staticmethod
    def extract_table_properties(parent: Document, element_type: str, property_name: str) -> Document:
        prompt =  """
        You are given a text string where columns are separated by comma representing either a single column, or multi-column table each new line is a new row.
        Instructions:
        1. Parse the table and return a flattened JSON object representing the key-value pairs of properties defined in the table.
        2. Do not return nested objects, keep the dictionary only 1 level deep. The only valid value types are numbers, strings, and lists.
        3. If you find multiple fields defined in a row, feel free to split them into separate properties.
        4. Use camelCase for the key names
        5. For fields where the values are in standard measurement units like miles, nautical miles, knots, celsius
        6. return only the json object between ``` 
        - include the unit in the key name and only set the numeric value as the value.
        - e.g. "Wind Speed: 9 knots" should become windSpeedInKnots: 9, "Temperature: 3°C" should become temperatureInC: 3
        """

        llm = OpenAI('gpt-4o-mini')
        output_property = property_name
        query_agent = LLMTextQueryAgent(prompt=prompt,  llm=llm, output_property=output_property, element_type=element_type, number_of_elements=1)
        doc =  query_agent.execute_query(parent)

        for ele in doc.elements:
            if ele.type == element_type and output_property in ele.properties.keys():
                try: 
                    jsonstring_llm = ele.properties.get(output_property)
                    logging.error(jsonstring_llm)
                    json_string = ExtractKeyValuePair.extract_parent_json(jsonstring_llm)
                    keyValue = json.loads(json_string)
                    ele.properties[output_property] = keyValue
                except Exception as e: 
                    logging.error(str(e))
        return doc
                    
if __name__ == "__main__":
    print('~~~~~~~~~~~~~~~~~~~~~')
    # json_string = 'json\n {\n "aircraftMake": "MARC JONES",\n"registration": "N512P",\n "modelSeries": "PITTS MODEL 12",\n"aircraftCategory": "Airplane",\n    "amateurBuilt": "",\n    "operator": "M12 AVIATION LLC",\n "operatingCertificatesHeld": "None"\n}\n'
    # processed_dict = ExtractKeyValuePair.extract_parent_json(json_string)
    # print(processed_dict)
    # print(json.loads(processed_dict))
    # print(type(json.loads(processed_dict)))

    def pickle_doc(doc: Document) -> bytes:
        return pickle.dumps(doc)


    def pickle_name(doc: Document, extension=None):
        return str(doc.doc_id) + ".pickle"


    def unpickle_doc(pdoc: Document) -> list[Document]:
        doc = pickle.loads(pdoc.binary_representation)
        return [doc]
    
    

    # # path = TEST_DIR / "resources/data/pdfs/Transformer.pdf"
    # from sycamore.utils.aryn_config import ArynConfig
    # s3_path = "s3://aryn-public/ntsb/59.pdf"
    # aryn_api_key = ArynConfig.get_aryn_api_key('/home/ec2-user/GIT31Jul/sycamore/notebooks/arynconfig.yaml')
    context = sycamore.init()
    # docs = (
    #     context.read.binary(s3_path, binary_format="pdf")
    #     .partition(ArynPartitioner(aryn_api_key= aryn_api_key, extract_table_structure=True, use_ocr=True, extract_images=True))

    #     .extract_key_value_pair(  element_type = 'table', property_name = 'llm_response')
    # )
    # docs = docs.filter_elements(lambda e:e.type=='table').assign_doc_properties(element_type = 'table', property_name = 'llm_response' )
    pickle_root = "/home/ec2-user/GIT31Jul/sycamore/notebooks/tmp"
    # docs.write.files(pickle_root, doc_to_bytes_fn=pickle_doc, filename_fn=pickle_name)

    pickled_docset = context.read.binary(str(pickle_root), binary_format="pickle")
    ds = pickled_docset.flat_map(unpickle_doc)
    from sycamore.transforms import LocationStandardizer
    from sycamore.transforms import DateTimeStandardizer

    loc_standardizer = LocationStandardizer()
    date_standardizer = DateTimeStandardizer()
    # ds.show()
    docset2 = (

        ds    
        # Normalize values
        .standardize(date_standardizer, path=['properties','entity','dateTime'])
        .standardize(loc_standardizer, path=['properties','entity','location'] )

    )
    docset2.show()


