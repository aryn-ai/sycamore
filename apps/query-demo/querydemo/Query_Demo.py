import contextlib
import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import boto3
import marko
from marko.md_renderer import MarkdownRenderer

import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from streamlit_pdf_viewer import pdf_viewer
import sycamore
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from util import get_opensearch_indices, generate_plan, run_plan, show_dag


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

EXAMPLE_QUERIES = [
    "How many incidents were there in Washington in 2023?",
    "Show me incidents involving tail number N4811E",
    "What was the breakdown of aircraft types for incidents with substantial damage?",
    "Show me accident ERA23LA153",
]


class MDRenderer(MarkdownRenderer):
    def __init__(self):
        super().__init__()

    def render_link(self, element: marko.inline.Link) -> str:
        if element.dest.startswith("s3://"):
            # Replace S3 links with a link to the demo instance.
            # (This is somewhat of a temporary hack until we build out better UI for this.)
            element.dest = element.dest.replace("s3://", "https://luna-demo.dev.aryn.ai/doc/")
        return super().render_link(element)


def rewrite_markdown(text: str):
    md = marko.Markdown(renderer=MDRenderer)
    doc = md.parse(text)
    return md.render(doc)


class ChatMessageExtra:
    def __init__(self, name: str, content: Any):
        self.name = name
        self.content = content


class ChatMessage:
    def __init__(self, message: Optional[Dict[str, Any]] = None, extras: Optional[List[ChatMessageExtra]] = None):
        self.timestamp = datetime.datetime.now()
        self.message = message or {}
        self.extras = extras or []

    def chat_message(self):
        return st.chat_message(self.message.get("role", "assistant"))

    def show(self):
        if self.message.get("content"):
            content = self.message.get("content")
            st.write(rewrite_markdown(content))
        for extra in self.extras:
            with st.expander(extra.name):
                st.write(extra.content)

    def to_dict(self) -> Optional[Dict[str, Any]]:
        return self.message


st.title("Sycamore NTSB Query Demo")

TOOLS: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "queryDataSource",
            "description": """Run a query against the backend data source. The query should be a natural language 
            query and can represent operations such as filters, aggregations, sorting, formatting,
            and more. This function should only be called when new data is needed; if the data
            is already available in the message history, it should be used directly.""",
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
def query_data_source(query: str, index: str) -> Tuple[str, LogicalPlan]:
    """Run a query against a back end data source."""
    sqclient = SycamoreQueryClient(
        s3_cache_path=st.session_state.s3_cache_path if st.session_state.use_cache else None,
    )
    with st.spinner("Generating plan..."):
        plan = generate_plan(sqclient, query, index)
    with st.spinner("Running query plan..."):
        st.session_state.query_id, result = run_plan(sqclient, plan)
    return str(result), plan


def do_query(prompt: str):
    user_message = ChatMessage({"role": "user", "content": prompt})
    st.session_state.messages.append(user_message)
    with user_message.chat_message():
        user_message.show()

    assistant_message = ChatMessage()
    query_plan = None
    with assistant_message.chat_message():
        while True:
            with st.spinner("Running query..."):
                response = openai_client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[m.to_dict() for m in st.session_state.messages],
                    tools=TOOLS,
                    tool_choice="auto",
                )
            response_dict = response.choices[0].message.to_dict()
            assistant_message.message = response_dict
            st.session_state.messages.append(assistant_message)

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                tool_call_id = tool_calls[0].id
                tool_function_name = tool_calls[0].function.name
                tool_args = json.loads(tool_calls[0].function.arguments)
                if tool_function_name == "queryDataSource":
                    tool_query = tool_args["query"]
                    tool_response, query_plan = query_data_source(tool_query, OPENSEARCH_INDEX)
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
                st.session_state.messages.append(tool_response_message)
                assistant_message = ChatMessage()

            else:
                # No function call was made.
                if query_plan:
                    assistant_message.extras.append(ChatMessageExtra("Query plan", query_plan))
                assistant_message.show()
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

for msg in st.session_state.messages:
    if msg.message.get("role") not in ["user", "assistant"]:
        continue
    if msg.message.get("content") is None:
        continue
    with msg.chat_message():
        msg.show()

if not st.session_state.messages:
    run_query = None
    for query in EXAMPLE_QUERIES:
        if st.button(query):
            run_query = query
    if run_query:
        do_query(run_query)

if prompt := st.chat_input("Ask me anything"):
    do_query(prompt)