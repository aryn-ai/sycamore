# This is the main page of the Sycamore Query UI.

import argparse
import time

import queryui.util as util
from queryui.configuration import get_sycamore_query_client
import queryui.ntsb as ntsb

import streamlit as st
from streamlit_ace import st_ace

from sycamore import ExecMode
from sycamore.executor import sycamore_ray_init
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan

PLANNER_EXAMPLES = ntsb.PLANNER_EXAMPLES


def generate_code(client: SycamoreQueryClient, plan: LogicalPlan) -> str:
    _, code = client.run_plan(plan, dry_run=True)
    return code


def show_schema(_client: SycamoreQueryClient, index: str):
    schema = util.get_schema(_client, index)
    table_data = []
    for key, values in schema.items():
        table_data.append([key] + list(values))
    with st.expander(f"Schema for index `[{index}]`"):
        st.dataframe(table_data)


@st.fragment
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
            if st.session_state.trace_dir:
                st.subheader("Traces", divider="blue")
                util.show_query_traces(st.session_state.trace_dir, st.session_state.query_id)


def run_query():
    """Run the given query."""
    client = get_sycamore_query_client(
        s3_cache_path=st.session_state.llm_cache_dir,
        trace_dir=st.session_state.trace_dir,
        cache_dir=st.session_state.cache_dir,
        exec_mode=ExecMode.LOCAL if st.session_state.local_mode else ExecMode.RAY,
    )
    with st.spinner("Generating plan..."):
        t1 = time.time()
        plan = util.generate_plan(client, st.session_state.query, st.session_state.index, examples=PLANNER_EXAMPLES)
        t2 = time.time()
    with st.expander("Query plan"):
        st.write(f"Generated plan in :blue[{t2 - t1:.2f}] seconds.")
        st.write(plan.model_dump(serialize_as_any=True))

    code = generate_code(client, plan)
    show_code(code)

    if not st.session_state.plan_only:
        with st.spinner("Running query..."):
            t1 = time.time()
            st.session_state.query_id, result = util.run_plan(client, plan)
            t2 = time.time()
            st.write(f"Ran query in :blue[{t2 - t1:.2f}] seconds.")

            result_str = util.result_to_string(result)
        st.write(f"Query ID: `{st.session_state.query_id}`\n")
        st.subheader("Result", divider="rainbow")
        st.markdown(result_str, unsafe_allow_html=True)

        if st.session_state.trace_dir:
            with st.expander("Query trace"):
                util.show_query_traces(st.session_state.trace_dir, st.session_state.query_id)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--local-mode", action="store_true", help="Enable Sycamore local execution mode.")
    argparser.add_argument(
        "--index", help="OpenSearch index name to use. If specified, only this index will be queried."
    )
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

    if "local_mode" not in st.session_state:
        st.session_state.local_mode = args.local_mode

    if "trace_dir" not in st.session_state:
        st.session_state.trace_dir = args.trace_dir

    if not args.local_mode:
        sycamore_ray_init(address="auto")
    client = get_sycamore_query_client(exec_mode=ExecMode.LOCAL if args.local_mode else ExecMode.RAY)

    st.title("Sycamore Query")
    st.write(f"Query cache dir: `{st.session_state.cache_dir}`")
    st.write(f"LLM cache dir: `{st.session_state.llm_cache_dir}`")
    st.write(f"Trace dir: `{st.session_state.trace_dir}`")

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
        show_schema(client, st.session_state.index)
        with st.form("query_form"):
            st.text_input("Query", key="query")
            col1, col2, col3 = st.columns(3)
            with col1:
                submitted = st.form_submit_button("Run query")
            with col2:
                st.toggle("Plan only", key="plan_only", value=False)
            with col3:
                st.toggle("Use Ray", key="use_ray", value=True)

        if submitted:
            run_query()


if __name__ == "__main__":
    main()
