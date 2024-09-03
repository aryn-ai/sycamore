import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import boto3
import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from streamlit_pdf_viewer import pdf_viewer
import sycamore
from sycamore.query.client import SycamoreQueryClient


from util import get_opensearch_indices, generate_plan, run_plan, show_dag


class ChatMessage:
    def __init__(self, message: Dict[str, Any]):
        self.timestamp = datetime.datetime.now()
        self.message = message

    def show(self):
        # Skip messages that have no content, e.g., tool call messages.
        if not self.message.get("content"):
            return
        # Skip messages that are not from the user or assistant, e.g., tool responses.
        if not self.message.get("role") in ["user", "assistant"]:
            return
        with st.chat_message(self.message["role"]):
            st.write(self.message['content'])

    def to_dict(self):
        return self.message


OPENSEARCH_INDEX = "const_ntsb"
OS_CONFIG = {"search_pipeline": "hybrid_pipeline"}
OS_CLIENT_ARGS = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "http_compress": True,
    "http_auth": ("admin", "admin"),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120,
}


st.title("Sycamore Query Demo")

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
                },
                "required": ["query"],
            },
        },
    },
]


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse an S3 path into a bucket and key."""
    s3_path = s3_path.replace("s3://", "")
    bucket, key = s3_path.split("/", 1)
    return bucket, key


def get_initial_documents():
    context = sycamore.init()
    docs = (
        context.read.opensearch(OS_CLIENT_ARGS, OPENSEARCH_INDEX)
        .filter(lambda doc: doc.properties.get("parent_id") is None)
        .take_all()
    )
    all_docs = {doc.properties.get("path"): doc for doc in docs}
    first_doc_path = sorted(all_docs.keys())[0]
    first_doc = all_docs[first_doc_path]
    show_document(first_doc)


@st.fragment
def show_document(doc: sycamore.data.Document):
    bucket, key = parse_s3_path(doc.properties.get("path"))
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read()
    if st.session_state.get("pagenum") is None:
        st.session_state.pagenum = 1

    with st.container(border=True):
        st.write(f"`{doc.properties.get('path')}`")
        tab1, tab2 = st.tabs(["PDF", "Metadata"])
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            if col1.button("First", use_container_width=True):
                st.session_state.pagenum = 1
            if col2.button("Prev", use_container_width=True):
                st.session_state.pagenum = max(1, st.session_state.pagenum - 1)
            if col3.button("Next", use_container_width=True):
                st.session_state.pagenum += 1
            col4.download_button(
                "Download", content, f"{doc.properties.get('path')}.pdf", "pdf", use_container_width=True
            )
            pdf_viewer(content, pages_to_render=[st.session_state.pagenum])

        with tab2:
            props = {k: v for k, v in doc.properties["entity"].items() if v is not None}
            st.dataframe(props)


@st.cache_data(show_spinner=False)
def query_data_source(query: str, index: str) -> str:
    """Run a query against a back end data source."""
    sqclient = SycamoreQueryClient(
        s3_cache_path=st.session_state.s3_cache_path if st.session_state.use_cache else None,
    )
    with st.spinner("Generating plan..."):
        plan = generate_plan(sqclient, query, index)
    with st.expander("View query plan"):
        show_dag(plan)

    with st.spinner("Running query..."):
        st.session_state.query_id, result = run_plan(sqclient, plan)
    return str(result)


def do_query():
    while True:
        with st.spinner("Running query..."):
            response = openai_client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[m.to_dict() for m in st.session_state.messages],
                tools=TOOLS,
                tool_choice="auto",
            )
        response_dict = response.choices[0].message.to_dict()
        response_message = ChatMessage(response_dict)
        st.session_state.messages.append(response_message)
        response_message.show()

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            tool_call_id = tool_calls[0].id
            tool_function_name = tool_calls[0].function.name
            tool_args = json.loads(tool_calls[0].function.arguments)
            if tool_function_name == "queryDataSource":
                query = tool_args["query"]
                tool_response = query_data_source(query, OPENSEARCH_INDEX)
            else:
                tool_response = f"Unknown tool: {tool_function_name}"

            tool_response_message = ChatMessage(
                {
                    "role": "tool",
                    "content": tool_response,
                    "tool_call_id": tool_call_id,
                    "name": tool_function_name,
                }
            )
            tool_response_message.show()
            st.session_state.messages.append(tool_response_message)
        else:
            # No function call was made.
            break


# Set OpenAI API key from Streamlit secrets
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "s3_cache_path" not in st.session_state:
    st.session_state.s3_cache_path = "s3://aryn-temp/llm_cache/luna/query-demo"

if "use_cache" not in st.session_state:
    st.session_state.use_cache = True

get_initial_documents()

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    #    if message.get("role") not in ["user", "assistant"]:
    #        continue
    msg.show()
#    with st.chat_message(message.get("role")):
#        st.write("Role: ", message.get("role"))
#        st.markdown(message.get("content", "<no content>"))

# Accept user input
if prompt := st.chat_input("Ask me anything"):
    user_message = ChatMessage({ "role": "user", "content": prompt })
    user_message.show()
    st.session_state.messages.append(user_message)
    do_query()
