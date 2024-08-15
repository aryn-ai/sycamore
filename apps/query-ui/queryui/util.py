import io
import os
import pickle
import zipfile
import pandas as pd
from typing import Any, Dict, Set, Tuple

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.logical_operator import LogicalOperator


@st.cache_data(show_spinner=False)
def get_schema(_client: SycamoreQueryClient, index: str) -> Dict[str, Tuple[str, Set[str]]]:
    return _client.get_opensearch_schema(index)


@st.cache_data(show_spinner=False)
def generate_plan(_client: SycamoreQueryClient, query: str, index: str) -> LogicalPlan:
    return _client.generate_plan(query, index, get_schema(_client, index))


@st.cache_data(show_spinner=False, hash_funcs={LogicalPlan: lambda x: hash(x.model_dump_json())})
def run_plan(_client: SycamoreQueryClient, plan: LogicalPlan) -> Tuple[str, Any]:
    return _client.run_plan(plan)


@st.cache_data(show_spinner=False)
def get_opensearch_indices() -> Set[str]:
    return {x for x in SycamoreQueryClient().get_opensearch_incides() if not x.startswith(".")}


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


@st.experimental_fragment
def show_query_traces(trace_dir: str, query_id: str):
    """Show the query traces in the given trace_dir."""
    trace_dir = os.path.join(trace_dir, query_id)
    for node_id in sorted(os.listdir(trace_dir)):
        data_list = []
        directory = os.path.join(trace_dir, node_id)
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                with open(f, "rb") as file:
                    try:
                        doc = pickle.load(file)
                    except EOFError:
                        doc = []

                    # For now, skip over MetadataDocuments.
                    if "doc_id" not in doc:
                        continue

                    if "properties" in doc:
                        for property in doc["properties"]:
                            if isinstance(doc["properties"][property], dict):
                                for nested_property in doc["properties"][property]:
                                    doc[".".join(["properties", property, nested_property])] = doc["properties"][
                                        property
                                    ][nested_property]
                            else:
                                doc[".".join(["properties", property])] = doc["properties"][property]
                        doc.pop("properties")
                    data_list.append(doc)

        df = pd.DataFrame(data_list)
        st.write(f"Docset after node {node_id} — {len(df)} documents")
        st.dataframe(df)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(trace_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                zf.write(file_path, os.path.relpath(file_path, trace_dir))
    st.download_button(
        label="Download Traces as ZIP",
        data=zip_buffer.getvalue(),
        file_name=f"traces_{query_id}.zip",
        mime="application/zip",
    )
