# This is a demo web UI for Sycamore Query based on Streamlit.
#
# To run: poetry run python -m streamlit run queryui/queryui.py

import os
import pickle
from typing import Any
import pandas as pd

import streamlit as st
from streamlit_ace import st_ace
from streamlit_agraph import agraph, Node, Edge, Config


from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.logical_operator import LogicalOperator


DEFAULT_S3_CACHE_PATH = "s3://aryn-temp/llm_cache/luna/ntsb"
DEFAULT_TRACE_DIR = "luna_traces"
DEFAULT_FULL_TRACE_DIR = f"{os.path.dirname(os.getcwd())}/query-ui/luna_traces"
BASE_PROPS = set(
    [
        "filename",
        "filetype",
        "page_number",
        "page_numbers",
        "links",
        "element_id",
        "parent_id",
        "_schema",
        "_schema_class",
        "entity",
    ]
)


def show_schema(container: Any, schema: dict[str, str]):
    # Make a table.
    table_data = []
    for key, value in schema.items():
        table_data.append([key, value])
    with container.expander("Schema"):
        st.dataframe(table_data)


def show_traces():
    """Show the traces in the given trace_dir."""
    base_dir = f"{DEFAULT_FULL_TRACE_DIR}/{st.session_state.query_id}"
    for id in sorted(os.listdir(f"{base_dir}")):

        # Initialize a list to hold the data
        data_list = []
        directory = f"{base_dir}/{id}"

        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                with open(f, "rb") as file:
                    doc = pickle.load(file)
                    doc_list = doc.properties["entity"]

                    for p in doc.properties:
                        if p not in BASE_PROPS and p in doc.properties:
                            doc_list[p] = doc.properties[p]

                    data_list.append(doc_list)

        df = pd.DataFrame(data_list)

        st.write(f"Docset after node {id} — {len(df)} documents")

        st.dataframe(df)


@st.experimental_fragment
def generate_code(client, plan):
    with st.spinner("Generating code..."):
        _, st.session_state.code = client.run_plan(plan, dry_run=True)
    with st.expander("View code"):
        st.session_state.code = st_ace(
            value=st.session_state.code,
            key="python",
            language="python",
            min_lines=20,
        )


def show_dag(plan: LogicalPlan):
    nodes = []
    edges = []
    for node in plan.nodes.values():
        assert isinstance(node, LogicalOperator)
        nodes.append(
            Node(
                id=node.node_id,
                label=f"{type(node).__name__}\n\n{node.description}",
                shape="box",
                color={
                    "background": "#404040",
                    "border": "#ff0000",
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


def run_query(query: str, index: str, plan_only: bool, do_trace: bool, use_cache: bool):
    """Run the given query."""
    st.session_state.trace_dir = None
    if do_trace:
        if not plan_only:
            st.session_state.trace_dir = DEFAULT_TRACE_DIR
            st.write(f"Writing execution traces to `{st.session_state.trace_dir}`")
        else:
            st.warning("Tracing currently not supported with plan only")
    if use_cache:
        st.write(f"Using cache at `{st.session_state.s3_cache_path}`")
    client = SycamoreQueryClient(
        trace_dir=st.session_state.trace_dir, s3_cache_path=st.session_state.s3_cache_path if use_cache else None
    )
    with st.spinner("Getting schema..."):
        schema = client.get_opensearch_schema(index)
    with st.spinner("Generating plan..."):
        plan = client.generate_plan(query, index, schema)
    with st.expander("View query plan"):
        show_dag(plan)
    if not plan_only:
        with st.spinner("Running query..."):
            st.session_state.query_id, result = client.run_plan(plan)
        st.write(f"Query ID `{st.session_state.query_id}`\n")
        st.subheader("Result", divider="rainbow")
        st.success(result)
        if do_trace:
            st.subheader("Traces", divider="blue")
            show_traces()

    else:
        generate_code(client, plan)
        if "code" in st.session_state and st.session_state.code:
            execute_button = st.button("Execute Code")
            if execute_button:
                code_locals: dict = {}
                try:
                    with st.spinner("Executing code..."):
                        exec(st.session_state.code, globals(), code_locals)
                except Exception as e:
                    st.exception(e)
                if code_locals and "result" in code_locals:
                    st.subheader("Result", divider="rainbow")
                    st.success(code_locals["result"])


client = SycamoreQueryClient()
indices = client.get_opensearch_incides()

st.title("Sycamore Query Demo")


with st.form("query_form"):
    st.text_input("Query", key="query")
    option = st.selectbox("Index", indices, key="index")
    schema_container = st.container()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        submitted = st.form_submit_button("Run query")
    with col2:
        plan_only = st.toggle("Plan only")
    with col3:
        do_trace = st.toggle("Capture traces")
    with col4:
        use_cache = st.toggle("Use cache")
    with st.expander("Advanced"):
        st.text_input("S3 cache path", key="s3_cache_path", value=DEFAULT_S3_CACHE_PATH)

if submitted:
    st.session_state.query_set = True
    show_schema(schema_container, client.get_opensearch_schema(st.session_state.index))
    run_query(st.session_state.query, st.session_state.index, plan_only, do_trace, use_cache)

elif "query_set" in st.session_state and st.session_state.query_set:
    show_schema(schema_container, client.get_opensearch_schema(st.session_state.index))
    run_query(st.session_state.query, st.session_state.index, plan_only, do_trace, use_cache)
