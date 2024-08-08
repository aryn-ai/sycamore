import os
from typing import Any, Dict, Set, Tuple

import streamlit as st
from streamlit_ace import st_ace
from streamlit_agraph import agraph, Node, Edge, Config

from util import show_query_traces

from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.logical_operator import LogicalOperator

DEFAULT_S3_CACHE_PATH = "s3://aryn-temp/llm_cache/luna/ntsb"


@st.cache_data(show_spinner=False)
def get_schema(_client: SycamoreQueryClient, index: str) -> Dict[str, Tuple[str, Set[str]]]:
    return _client.get_opensearch_schema(index)


@st.cache_data(show_spinner=False)
def generate_plan(_client: SycamoreQueryClient, query: str, index: str) -> LogicalPlan:
    return _client.generate_plan(query, index, get_schema(client, index))


@st.cache_data(show_spinner=False, hash_funcs={LogicalPlan: lambda x: hash(x.model_dump_json())})
def run_plan(_client: SycamoreQueryClient, plan: LogicalPlan) -> Tuple[str, Any]:
    return _client.run_plan(plan)


@st.cache_data(show_spinner=False)
def get_opensearch_indices() -> Set[str]:
    return {x for x in SycamoreQueryClient().get_opensearch_incides() if not x.startswith(".")}


def generate_code(client: SycamoreQueryClient, plan: LogicalPlan) -> str:
    st.session_state.query_id, code = client.run_plan(plan, dry_run=True)
    return code


def show_schema(_client: SycamoreQueryClient, index: str):
    schema = get_schema(_client, index)
    table_data = []
    for key, (value, _) in schema.items():
        table_data.append([key, value])
    with st.expander(f"Schema for index `[{index}]`"):
        st.dataframe(table_data)


@st.experimental_fragment
def show_code(code: str):
    with st.expander("View code"):
        code = st_ace(
            value=code,
            key="python",
            language="python",
            min_lines=20,
        )
        execute_button = st.button("Execute Code")
        if execute_button:
            code_locals: dict = {}
            try:
                with st.spinner("Executing code..."):
                    exec(code, globals(), code_locals)
            except Exception as e:
                st.exception(e)
            if code_locals and "result" in code_locals:
                st.subheader("Result", divider="rainbow")
                st.success(code_locals["result"])
            if st.session_state.do_trace:
                assert st.session_state.trace_dir
                st.subheader("Traces", divider="blue")
                show_query_traces(st.session_state.trace_dir, st.session_state.query_id)


def show_dag(plan: LogicalPlan):
    nodes = []
    edges = []
    for node in plan.nodes.values():
        assert isinstance(node, LogicalOperator)
        nodes.append(
            Node(
                id=node.node_id,
                label=f"[Node {node.node_id}] {type(node).__name__}\n\n{node.description}",
                shape="box",
                color={
                    "background": "#404040",
                    "border": "#5050f0",
                },
                font="14px arial white",
                chosen=False,
                margin=30,
            )
        )
    for node in plan.nodes.values():
        if node.dependencies:
            for dep in node.dependencies:
                edges.append(Edge(source=dep.node_id, target=node.node_id, color="#ffffff"))

    config = Config(
        width=700,
        height=500,
        directed=True,
        physics=False,
        hierarchical=True,
        direction="UD",
    )
    agraph(nodes=nodes, edges=edges, config=config)


def run_query():
    """Run the given query."""
    if st.session_state.do_trace:
        assert st.session_state.trace_dir
        st.write(f"Writing execution traces to `{st.session_state.trace_dir}`")
    if st.session_state.s3_cache_path:
        st.write(f"Using S3 cache at `{st.session_state.s3_cache_path}`")

    client = SycamoreQueryClient(
        trace_dir=st.session_state.trace_dir,
        s3_cache_path=st.session_state.s3_cache_path if st.session_state.use_cache else None,
    )
    with st.spinner("Generating plan..."):
        plan = generate_plan(client, st.session_state.query, st.session_state.index)
    with st.expander("View query plan"):
        show_dag(plan)

    code = generate_code(client, plan)
    show_code(code)

    st.write(f"Query ID `{st.session_state.query_id}`\n")

    if not st.session_state.plan_only:
        with st.spinner("Running query..."):
            st.session_state.query_id, result = run_plan(client, plan)
        st.subheader("Result", divider="rainbow")
        st.success(result)
        if st.session_state.do_trace:
            assert st.session_state.trace_dir
            st.subheader("Traces", divider="blue")
            show_query_traces(st.session_state.trace_dir, st.session_state.query_id)


st.title("Sycamore Query")


if "trace_dir" not in st.session_state:
    st.session_state.trace_dir = os.path.join(os.getcwd(), "traces")


client = SycamoreQueryClient()
st.selectbox("Index", get_opensearch_indices(), key="index")
show_schema(client, st.session_state.index)


with st.form("query_form"):
    st.text_input("Query", key="query")
    schema_container = st.container()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        submitted = st.form_submit_button("Run query")
    with col2:
        st.toggle("Plan only", key="plan_only", value=False)
    with col3:
        st.toggle("Capture traces", key="do_trace", value=True)
    with col4:
        st.toggle("Use cache", key="use_cache", value=True)
    with st.expander("Advanced"):
        st.text_input("S3 cache path", key="s3_cache_path", value=DEFAULT_S3_CACHE_PATH)
        st.session_state.trace_dir = st.text_input("Trace directory", value=st.session_state.trace_dir)


if submitted:
    run_query()
