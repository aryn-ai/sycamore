import json
import os
from typing import List

import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from configuration import get_sycamore_query_client

from util import get_opensearch_indices, generate_plan, run_plan, show_dag

client = get_sycamore_query_client()

st.title("Sycamore Query Chat")

TOOLS: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "queryDataSource",
            "description": "Run a query against a back end data source",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query",
                    },
                    "index": {
                        "type": "string",
                        "description": "The name of the index that the user wants to query",
                    },
                },
                "required": ["query", "index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getDataSourceIndices",
            "description": "Return the list of data source indices available for querying",
        },
    },
]


@st.cache_data(show_spinner=False)
def get_data_source_indices() -> str:
    """Return the list of data source indices available for querying."""
    return ", ".join(list(get_opensearch_indices()))


@st.cache_data(show_spinner=False)
def query_data_source(query: str, index: str) -> str:
    """Run a query against a back end data source."""

    if st.session_state.s3_cache_path:
        st.write(f"Using S3 cache at `{st.session_state.s3_cache_path}`")

    client = get_sycamore_query_client(
        s3_cache_path=st.session_state.s3_cache_path if st.session_state.use_cache else None
    )
    with st.spinner("Generating plan..."):
        plan = generate_plan(client, query, index)
    with st.expander("View query plan"):
        show_dag(plan)

    with st.spinner("Running query..."):
        st.session_state.query_id, result = run_plan(client, plan)
    return str(result)


def do_query():
    while True:
        response = client.chat.completions.create(
            model=st.session_state["openai_model"], messages=st.session_state.messages, tools=TOOLS, tool_choice="auto"
        )
        response_message = response.choices[0].message
        st.session_state.messages.append(response_message.to_dict())

        tool_calls = response_message.tool_calls
        if tool_calls:
            tool_call_id = tool_calls[0].id
            tool_function_name = tool_calls[0].function.name
            tool_args = json.loads(tool_calls[0].function.arguments)
            if tool_function_name == "queryDataSource":
                query = tool_args["query"]
                index = tool_args["index"]
                tool_response = query_data_source(query, index)
            elif tool_function_name == "getDataSourceIndices":
                tool_response = get_data_source_indices()
            else:
                tool_response = f"Unknown tool: {tool_function_name}"

            st.session_state.messages.append(
                {
                    "role": "tool",
                    "content": tool_response,
                    "tool_call_id": tool_call_id,
                    "name": tool_function_name,
                }
            )
        else:
            # No function call was made.
            final_response = response_message.content
            st.write(final_response)
            break


# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "s3_cache_path" not in st.session_state:
    st.session_state.s3_cache_path = "s3://aryn-temp/llm_cache/luna/ntsb"

if "use_cache" not in st.session_state:
    st.session_state.use_cache = True

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message.get("role") not in ["user", "assistant"]:
        continue
    with st.chat_message(message.get("role")):
        st.markdown(message.get("content", "<no content>"))

# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        do_query()
