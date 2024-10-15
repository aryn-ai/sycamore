import datetime
import os

import streamlit as st

from util import show_query_traces


def show_traces(trace_dir: str):
    # Get all directories in the trace directory.
    query_dirs = [d for d in os.listdir(trace_dir) if os.path.isdir(os.path.join(trace_dir, d))]
    query_dirs_dt = [
        (d, datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(trace_dir, d)))) for d in query_dirs
    ]
    query_dirs_dt.sort(key=lambda x: x[1], reverse=True)

    def trace_summary(query_dir):
        return f"{query_dir[0]} ({query_dir[1]})"

    query_id = st.selectbox(
        "Select a trace",
        query_dirs_dt,
        format_func=trace_summary,
        placeholder="Select a trace",
        label_visibility="collapsed",
    )
    if query_id:
        st.header(f"Query `{query_id[0]}`")
        st.subheader(query_id[1], divider=True)
        show_query_traces(trace_dir, query_id[0])


st.title("Trace Viewer")

st.session_state.trace_dir = st.text_input("Trace directory", value=st.session_state.get("trace_dir", ""))
show_traces(st.session_state.trace_dir)
