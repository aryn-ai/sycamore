from sycamore.data import Document, Table
from sycamore.functions import HuggingFaceTokenizer
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms.extract_schema import OpenAISchemaExtractor, OpenAIPropertyExtractor
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
#from sycamore.transforms.merge_elements import GreedySectionMerger
from sycamore.transforms.partition import ArynPartitioner, SycamorePartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.summarize_images import SummarizeImages
from sycamore.utils.pdf_utils import show_pages

from sycamore.data import BoundingBox, Document, Element, TableElement
from sycamore.functions.document import split_and_convert_to_image, DrawBoxes
import sycamore
import time
from pathlib import Path
import pickle
from dateutil import parser

from opensearchpy import OpenSearch

from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.data import OpenSearchQuery
from sycamore.utils.time_trace import timetrace

import json

import os
import sys 

import PIL.Image

from io import BytesIO

import pprint

def pickle_doc(doc: Document) -> bytes:
    return pickle.dumps(doc)

def pickle_name(doc: Document, extension=None):
    return str(doc.doc_id) + ".pickle"

def unpickle_doc(pdoc: Document) -> list[Document]:
    doc = pickle.loads(pdoc.binary_representation)
    return [doc]

ctx = sycamore.init()

from sycamore.utils.aryn_config import ArynConfig,_DEFAULT_PATH
s3_path = "s3://aryn-public/ntsb/59.pdf"
llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
# llm = OpenAI(OpenAIModels.GPT_4O.value)
tokenizer = HuggingFaceTokenizer("thenlper/gte-small")

# docset = (
#     ctx.read.binary(s3_path, binary_format="pdf")

#     # Partition and extract tables and images
#     .partition(partitioner=ArynPartitioner(
#         aryn_api_key = ArynConfig.get_aryn_api_key('GIT31Jul/sycamore/notebooks/arynconfig.yaml'),
#         extract_table_structure=True, 
#         use_ocr=True, 
#         extract_images=True))

#     # Summarize each image element
#     .transform(SummarizeImages)
# )



pickled_docset = ctx.read.binary(str('/home/ec2-user/GIT31Jul/sycamore/notebooks/tmp2/e5b72d9e-4f88-11ef-966d-0eae5337fe69.pickle'), binary_format="pickle")
ds = pickled_docset.flat_map(unpickle_doc)

# def customFunc(d):
#     if 'table' in d and isinstance(d['table'], Table):
#         d.text_representation = d['table'].to_csv()
#     return d

# # filter page
# # ds.filter_elements(lambda el: isinstance(el, TableElement))

# ds_table = ds.map(customFunc)
# doc = ds_table.take_all()
# for i in doc[0].elements:
#     # print(i)
#     if i.type == "table":
#         print(i.text_representation)

from sycamore.transforms.llm_query import LLMTextQueryAgent,LLMQuery
from sycamore.functions.elements import filter_elements
from sycamore.llms import OpenAI

llm = OpenAI('gpt-4o-mini')

prompt  = """
    You are given a text string where columns are separated by comma representing either a single column, or multi-column table each new line is a new row.
    Instructions:
    1. Parse the table and return a flattened JSON object representing the key-value pairs of properties defined in the table.
    2. Do not return nested objects, keep the dictionary only 1 level deep. The only valid value types are numbers, strings, and lists.
    3. If you find multiple fields defined in a row, feel free to split them into separate properties.
    4. Use camelCase for the key names
    5. For fields where the values are in standard measurement units like miles, nautical miles, knots, celsius
    6. return only the json object
       - include the unit in the key name and only set the numeric value as the value.
       - e.g. "Wind Speed: 9 knots" should become windSpeedInKnots: 9, "Temperature: 3°C" should become temperatureInC: 3
    """
# llm_query_agent = LLMTextQueryAgent(prompt=prompt, llm = llm,per_element=True, element_type = 'table', number_of_elements=10)
# # ds_llm = ds_table.filter_elements(lambda d:d.type=='table').filter_elements(lambda d:d.properties['page_number']==1).llm_query(query_agent=llm_query_agent)

# # llm_query_agent = LLMTextQueryAgent(prompt=prompt, llm = llm,per_element=True, element_type = 'table', number_of_elements=10)
# ds_llm = ds.llm_query(query_agent=llm_query_agent)

# docs = ds_llm.filter_elements(lambda d:d.type=='table')


# # docs.show()
# ds_llm.write.files('/home/ec2-user/GIT31Jul/sycamore/notebooks/llmwrite', doc_to_bytes_fn=pickle_doc, filename_fn=pickle_name)

# ds_llm.filter_elements(lambda d:d.type=='table').show()
pickled_docset = ctx.read.binary(str('/home/ec2-user/GIT31Jul/sycamore/notebooks/llmwrite'), binary_format="pickle")
ds = pickled_docset.flat_map(unpickle_doc)


ds.filter_elements(lambda d:d.type=='table').show()

# element_type = 'table'
# element_idx = 0
# elementPro = "llm_response"
# def parse_json_garbage(s):
#     s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
#     try:
#         return json.loads(s)
#     except json.JSONDecodeError as e:
#         return json.loads(s[:e.pos])
    
# for doc in ds.take_all():
#     element_count = 0
#     for e in doc.elements:
#         if e.type == element_type:
#             if element_count==element_idx:
#                 # print(e)
#                 e.properties.update(parse_json_garbage(e.properties.get(elementPro)))
#                 doc.properties["entity"] = e.properties.copy()
#             element_count +=1


from sycamore.transforms import assign_doc_properties
ds = ds.assign_doc_properties('table')
ds.show(limit=2, show_elements = False)

ds1 = ds.filter_elements(lambda d:d.properties['page_number']==1)
ds1.show()



