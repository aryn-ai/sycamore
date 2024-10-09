# This file contains the Chat demo portion of the Sycamore Query UI, which uses an LLM-based
# agent to formulate and render responses from Sycamore queries.

import argparse
import json
import os
from typing import Any, List, Optional, Tuple

from queryui.chat import ChatMessage, ChatMessageExtra, ChatMessageTraces, MDX_SYSTEM_PROMPT
import queryui.util as util
import queryui.ntsb as ntsb

from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
import requests
import streamlit as st
import sycamore
from sycamore.data import OpenSearchQuery
from sycamore import ExecMode
from sycamore.executor import sycamore_ray_init
from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.query.client import SycamoreQueryClient

# The OpenAI model used for the chat agent.
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
OPENAI_MODEL = "gpt-4o"

# We assume that OpenSearch is running locally on port 9200.
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

# These are application- and dataset-specific parameters that can be adjusted as needed.
# The defaults are in ntsb.py.
SYSTEM_PROMPT = ntsb.SYSTEM_PROMPT
PLANNER_EXAMPLES = ntsb.PLANNER_EXAMPLES
WELCOME_MESSAGE = ntsb.WELCOME_MESSAGE

# The set of tools passed to the LLM. In this case, we only pass a queryDataSource tool, which is used
# to run a Sycamore query.
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


# The set of tools passed to the LLM when "RAG only" mode is chosen. In this case, we only pass
# a queryDataSource tool, which performs a vector search lookup, leaving the rest of the RAG
# processing to the agent itself.
TOOLS_RAG: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "queryDataSource",
            "description": """Run a query against the backend data source. The query should be a
            natural language query. This function should only be called when new data is needed;
            if the data is already available in the message history, it should be used directly.""",
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


def get_embedding_model_id() -> str:
    """Get the embedding model ID from the OpenSearch instance. Used for RAG queries."""
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {"name": "all-MiniLM-L6-v2"},
                    },
                    {
                        "term": {"model_config.model_type": "bert"},
                    },
                ],
            },
        },
    }
    with requests.get(
        "https://localhost:9200/_plugins/_ml/models/_search", json=query, verify=False, timeout=60
    ) as resp:
        res = json.loads(resp.text)
        return res["hits"]["hits"][0]["_id"]


def do_rag_query(query: str, index: str) -> str:
    """Uses OpenSearch to perform a RAG query."""
    embedding_model_id = get_embedding_model_id()
    search_pipeline = "hybrid_rag_pipeline"
    llm = "gpt-4o"
    rag_query = OpenSearchQuery()
    rag_query["index"] = index
    rag_query["query"] = {
        "_source": {"excludes": ["embedding"]},
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"text_representation": query}},
                    {
                        "neural": {
                            "embedding": {
                                "query_text": query,
                                "model_id": embedding_model_id,
                                "k": 100,
                            }
                        }
                    },
                ]
            }
        },
        "size": 20,
    }

    # RAG params
    rag_query["params"] = {"search_pipeline": search_pipeline}
    rag_query["query"]["ext"] = {
        "generative_qa_parameters": {
            "llm_question": query,
            "context_size": 10,
            "llm_model": llm,
        }
    }

    with st.expander("RAG query"):
        st.write(rag_query)

    with st.spinner("Running RAG query..."):
        osq = OpenSearchQueryExecutor(OS_CLIENT_ARGS)
        rag_result = osq.query(rag_query)["result"]
    return rag_result


def query_data_source(query: str, index: str) -> Tuple[Any, Optional[Any], Optional[str]]:
    """Run a Sycamore or RAG query.

    Returns a tuple of (query_result, query_plan, query_id).
    """

    if st.session_state.rag_only:
        return do_rag_query(query, index), None, None
    else:
        sqclient = SycamoreQueryClient(
            s3_cache_path=st.session_state.llm_cache_dir,
            trace_dir=st.session_state.trace_dir,
            cache_dir=st.session_state.cache_dir,
            sycamore_exec_mode=ExecMode.LOCAL if st.session_state.local_mode else ExecMode.RAY,
        )
        with st.spinner("Generating plan..."):
            plan = util.generate_plan(sqclient, query, index, examples=PLANNER_EXAMPLES)
            print(f"Generated plan:\n{plan}\n")
            # No need to show the prompt used in the demo.
            plan.llm_prompt = None
            with st.expander("Query plan"):
                st.write(plan)
        with st.spinner("Running Sycamore query..."):
            print("Running plan...")
            query_id, result = util.run_plan(sqclient, plan)
            print(f"Ran query ID: {query_id}")
        return result, plan, query_id


def show_messages():
    """Show all user and assistant messages."""
    for msg in st.session_state.messages:
        if msg.message.get("role") not in ["user", "assistant"]:
            continue
        if msg.message.get("content") is None:
            continue
        msg.show()


def do_query():
    """Run a query based on the user's input."""

    prompt = st.session_state.user_query
    st.session_state.user_query = None
    print(f"User query: {prompt}")
    user_message = ChatMessage({"role": "user", "content": prompt})
    st.session_state.messages.append(user_message)
    user_message.show()

    assistant_message = None
    query_plan = None
    query_id = None
    tool_response_str = None
    with st.chat_message("assistant"):
        # We loop here because tool calls require re-invoking the LLM.
        while True:
            system_prompt = {
                "role": "system",
                "content": SYSTEM_PROMPT + MDX_SYSTEM_PROMPT,
            }
            messages = [system_prompt] + [m.to_dict() for m in st.session_state.messages]
            with st.spinner("Running LLM query..."):
                response = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    tools=TOOLS_RAG if st.session_state.rag_only else TOOLS,
                    tool_choice="auto",
                )
            response_dict = response.choices[0].message.to_dict()
            assistant_message = ChatMessage(response_dict)
            st.session_state.messages.append(assistant_message)

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                tool_call_id = tool_calls[0].id
                tool_function_name = tool_calls[0].function.name
                tool_response_str = ""
                query_plan = None
                query_id = None
                # Try to catch any errors that might corrupt the message history here.
                try:
                    tool_args = json.loads(tool_calls[0].function.arguments)
                    if tool_function_name == "queryDataSource":
                        tool_query = tool_args["query"]
                        tool_response, query_plan, query_id = query_data_source(tool_query, st.session_state.index)
                    else:
                        tool_response = f"Unknown tool: {tool_function_name}"

                    with st.spinner("Running Sycamore query..."):
                        if isinstance(tool_response, str):
                            # We got a straight string response from the query plan, which means we can
                            # feed it back to the LLM directly.
                            tool_response_str = tool_response
                        elif isinstance(tool_response, sycamore.docset.DocSet):
                            # We got a DocSet.
                            # Note that this can be slow because the .take()
                            # actually runs the query.
                            tool_response_str = util.docset_to_string(tool_response, html=False)
                        else:
                            # Fall back to string representation.
                            tool_response_str = str(tool_response)
                        if not tool_response_str:
                            tool_response_str = "No results found for your query."
                except Exception as e:
                    st.error(f"Error running Sycamore query: {e}")
                    print(f"Error running Sycamore query: {e}")
                    # Print stack trace.
                    import traceback

                    traceback.print_exc()
                    tool_response_str = f"There was an error running your query: {e}"
                finally:
                    print(f"\nTool response: {tool_response_str}")
                    tool_response_message = ChatMessage(
                        {
                            "role": "tool",
                            "content": tool_response_str,
                            "tool_call_id": tool_call_id,
                            "name": tool_function_name,
                        }
                    )
                    st.session_state.messages.append(tool_response_message)
                    with st.expander("Sycamore query result"):
                        st.write(f"```{tool_response_str}```")

            else:
                # No function call was made.
                assistant_message.show_content()
                if query_plan:
                    assistant_message.before_extras.append(ChatMessageExtra("Query plan", query_plan))
                if tool_response_str:
                    assistant_message.before_extras.append(
                        ChatMessageExtra("Sycamore query result", f"```{tool_response_str}```")
                    )
                if query_id:
                    cmt = ChatMessageTraces("Query trace", query_id)
                    with st.expander("Query trace"):
                        cmt.show()
                    assistant_message.after_extras.append(cmt)
                break


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--index", help="OpenSearch index name to use. If specified, only this index will be queried."
    )
    argparser.add_argument("--local-mode", action="store_true", help="Enable Sycamore local execution mode.")
    argparser.add_argument("--title", type=str, help="Title text.")
    argparser.add_argument("--cache-dir", type=str, help="Query execution cache dir.")
    argparser.add_argument("--llm-cache-dir", type=str, help="LLM query cache dir.")
    argparser.add_argument("--trace-dir", type=str, help="Directory to store query traces.")
    args = argparser.parse_args()

    if "index" not in st.session_state:
        st.session_state.index = args.index

    if "cache_dir" not in st.session_state:
        st.session_state.cache_dir = args.cache_dir

    if "llm_cache_dir" not in st.session_state:
        st.session_state.llm_cache_dir = args.llm_cache_dir

    if "trace_dir" not in st.session_state:
        st.session_state.trace_dir = args.trace_dir

    if "use_cache" not in st.session_state:
        st.session_state.use_cache = True

    if "local_mode" not in st.session_state:
        st.session_state.local_mode = args.local_mode

    if "next_message_id" not in st.session_state:
        st.session_state.next_message_id = 0

    if "user_query" not in st.session_state:
        st.session_state.user_query = None

    if "messages" not in st.session_state:
        st.session_state.messages = [ChatMessage({"role": "assistant", "content": WELCOME_MESSAGE})]

    if "trace_dir" not in st.session_state:
        st.session_state.trace_dir = os.path.join(os.getcwd(), "traces")

    if not args.local_mode:
        sycamore_ray_init(address="auto")
    st.title("Sycamore Query Chat")
    st.toggle("Use RAG only", key="rag_only")

    if not args.index:
        with st.spinner("Loading indices..."):
            try:
                st.session_state.indices = util.get_opensearch_indices()
            except Exception as e:
                st.error(f"Unable to load OpenSearch indices. Is OpenSearch running?\n\n{e}")
                return
        st.selectbox(
            "Index",
            util.get_opensearch_indices(),
            key="index",
            placeholder="Select an index",
            label_visibility="collapsed",
        )

    if st.session_state.index:
        show_messages()
        if prompt := st.chat_input("Ask me anything"):
            st.session_state.user_query = prompt

        while st.session_state.user_query is not None:
            do_query()


if __name__ == "__main__":
    main()
