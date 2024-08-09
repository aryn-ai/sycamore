from sycamore.llms import OpenAI, OpenAIModels
from sycamore.llms.prompts.default_prompts import LlmFilterMessagesPrompt
from sycamore.query.execution.operations import math_operation
from sycamore.query.execution.operations import summarize_data
from sycamore.functions.basic_filters import RangeFilter
from sycamore.query.execution.metrics import SycamoreQueryLogger
from sycamore.utils.cache import S3Cache
import sycamore

context = sycamore.init()

# Get all the incident reports
os_client_args = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "http_compress": True,
    "http_auth": ("admin", "admin"),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120,
}
output_0 = context.read.opensearch(os_client_args=os_client_args, index_name="const_ntsb")

# Filter to only include incidents in the first half of January 2023
output_1 = output_0.filter(
    f=RangeFilter(field="properties.entity.day", start="2023-01-01", end="2023-01-15", date=True), **{"name": "1"}
)

# Filter to only include environmentally caused incidents
prompt = LlmFilterMessagesPrompt(
    filter_question="Was this incident caused by environmental factors?"
).get_messages_dict()
output_2 = output_1.llm_filter(
    llm=OpenAI(OpenAIModels.GPT_4O.value, cache=S3Cache("s3://aryn-temp/llm_cache/luna/ntsb")),
    new_field="_autogen_LLMFilterOutput",
    prompt=prompt,
    field="text_representation",
    threshold=3,
    **{"name": "2"},
)

# Count the number of environmentally caused incidents
output_3 = output_2.count_distinct(
    field="properties.entity.accidentNumber",
    **{
        "write_intermediate_data": True,
        "intermediate_datasink": SycamoreQueryLogger,
        "intermediate_datasink_kwargs": {
            "query_id": "d4469ff2-4a49-4fa1-a7a8-6b57199b0c32",
            "node_id": 3,
            "path": "/Users/tanviranade/Tanvi/sycamore/apps/query-ui/traces/d4469ff2-4a49-4fa1-a7a8-6b57199b0c32/",
            "makedirs": True,
            "verbose": True,
        },
        "name": "3",
    },
)

# Filter to only include incidents caused by wind
prompt = LlmFilterMessagesPrompt(filter_question="Was this incident caused by wind?").get_messages_dict()
output_4 = output_2.llm_filter(
    llm=OpenAI(OpenAIModels.GPT_4O.value, cache=S3Cache("s3://aryn-temp/llm_cache/luna/ntsb")),
    new_field="_autogen_LLMFilterOutput",
    prompt=prompt,
    field="text_representation",
    threshold=3,
    **{"name": "4"},
)

# Count the number of incidents caused by wind
output_5 = output_4.count_distinct(
    field="properties.entity.accidentNumber",
    **{
        "write_intermediate_data": True,
        "intermediate_datasink": SycamoreQueryLogger,
        "intermediate_datasink_kwargs": {
            "query_id": "d4469ff2-4a49-4fa1-a7a8-6b57199b0c32",
            "node_id": 5,
            "path": "/Users/tanviranade/Tanvi/sycamore/apps/query-ui/traces/d4469ff2-4a49-4fa1-a7a8-6b57199b0c32/",
            "makedirs": True,
            "verbose": True,
        },
        "name": "5",
    },
)

# Divide the number of wind-caused incidents by the total number of environmentally caused incidents
output_6 = math_operation(val1=output_5, val2=output_3, operator="divide")

# Generate an English response to the question. Input 1 is the fraction of environmentally caused incidents due to wind in the first half of January 2023.
result = summarize_data(
    llm=OpenAI(OpenAIModels.GPT_4O.value, cache=S3Cache("s3://aryn-temp/llm_cache/luna/ntsb")),
    question="What fraction of environmentally caused incidents were due to wind in the first half of January 2023?",
    result_description="Generate an English response to the question. Input 1 is the fraction of environmentally caused incidents due to wind in the first half of January 2023.",
    result_data=[output_6],
    **{
        "write_intermediate_data": True,
        "intermediate_datasink": SycamoreQueryLogger,
        "intermediate_datasink_kwargs": {
            "query_id": "d4469ff2-4a49-4fa1-a7a8-6b57199b0c32",
            "node_id": 7,
            "path": "/Users/tanviranade/Tanvi/sycamore/apps/query-ui/traces/d4469ff2-4a49-4fa1-a7a8-6b57199b0c32/",
            "makedirs": True,
            "verbose": True,
        },
        "name": "7",
    },
)
