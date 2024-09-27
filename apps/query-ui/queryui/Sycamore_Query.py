import os

import streamlit as st
from streamlit_ace import st_ace

from sycamore.executor import sycamore_ray_init
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan

from configuration import get_sycamore_query_client
import util

if "EXTERNAL_RAY" in os.environ:
    # For not yet understood reasons, the ray processes will die when started under streamlit.
    # Which then crashes the streamlit app. Running ray externally and connecting to it fixes those problems.
    print("Configuring ray with auto address")
    sycamore_ray_init(address="auto")

# Streamlit is swallowing command line arguments. trying with -- didn't work for me.
# https://github.com/streamlit/streamlit/issues/337
DEFAULT_S3_CACHE_PATH = os.getenv("QUERY_CACHE", default="s3://aryn-temp/llm_cache/luna/ntsb")


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
            if st.session_state.do_trace:
                assert st.session_state.trace_dir
                st.subheader("Traces", divider="blue")
                util.show_query_traces(st.session_state.trace_dir, st.session_state.query_id)


def run_query():
    """Run the given query."""
    if st.session_state.do_trace:
        assert st.session_state.trace_dir
        st.write(f"Writing execution traces to `{st.session_state.trace_dir}`")
    if st.session_state.s3_cache_path:
        st.write(f"Using S3 cache at `{st.session_state.s3_cache_path}`")

    client = get_sycamore_query_client(
        s3_cache_path=st.session_state.s3_cache_path if st.session_state.use_cache else None,
        trace_dir=st.session_state.trace_dir,
    )
    with st.spinner("Generating plan..."):
        plan = util.generate_plan(client, st.session_state.query, st.session_state.index)
    with st.expander("Query plan"):
        st.write(plan.model_dump(serialize_as_any=True))

    code = generate_code(client, plan)
    show_code(code)

    if not st.session_state.plan_only:
        with st.spinner("Running query..."):
            st.session_state.query_id, result = util.run_plan(client, plan)
            result_str = util.result_to_string(result)
        st.write(f"Query ID `{st.session_state.query_id}`\n")
        st.subheader("Result", divider="rainbow")
        st.markdown(result_str, unsafe_allow_html=True)

        if st.session_state.do_trace:
            assert st.session_state.trace_dir
            st.subheader("Traces", divider="blue")
            util.show_query_traces(st.session_state.trace_dir, st.session_state.query_id)


st.title("Sycamore Query")


if "trace_dir" not in st.session_state:
    st.session_state.trace_dir = os.path.join(os.getcwd(), "traces")

if "index" not in st.session_state:
    st.session_state.index = None

client = get_sycamore_query_client()
st.selectbox("Index", util.get_opensearch_indices(), key="index")

if st.session_state.index:
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
