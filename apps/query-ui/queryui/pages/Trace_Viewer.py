import datetime
import os

import streamlit as st

from util import show_query_traces


def show_traces(trace_dir: str):
    st.write("Trace directory: ", trace_dir)
    # Get all directories in the trace directory.
    query_dirs = [d for d in os.listdir(trace_dir) if os.path.isdir(os.path.join(trace_dir, d))]
    query_dirs_dt = [
        (d, datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(trace_dir, d)))) for d in query_dirs
    ]
    query_dirs_dt.sort(key=lambda x: x[1], reverse=True)
    for query_id in query_dirs_dt:
        with st.expander(f"Query `{query_id[0]}` ({query_id[1]})", expanded=False):
            st.write(f"Query {query_id}")
            show_query_traces(trace_dir, query_id[0])


st.title("Trace Viewer")

with st.form("trace_form"):
    st.session_state.trace_dir = st.text_input("Trace directory", value=st.session_state.get("trace_dir", ""))
    submitted = st.form_submit_button("View traces")

if submitted:
    show_traces(st.session_state.trace_dir)
