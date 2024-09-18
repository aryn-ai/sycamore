import io
import logging
import os
import pickle
import zipfile
import pandas as pd
from typing import Any, Dict, List, Set, Tuple

import ray
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from sycamore.executor import _ray_logging_setup
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.logical_operator import LogicalOperator


def ray_init(**ray_args):
    if ray.is_initialized():
        return

    if "logging_level" not in ray_args:
        ray_args.update({"logging_level": logging.INFO})
    if "runtime_env" not in ray_args:
        ray_args["runtime_env"] = {}
    if "worker_process_setup_hook" not in ray_args["runtime_env"]:
        ray_args["runtime_env"]["worker_process_setup_hook"] = _ray_logging_setup
    ray.init(**ray_args)


def get_schema(_client: SycamoreQueryClient, index: str) -> Dict[str, Tuple[str, Set[str]]]:
    return _client.get_opensearch_schema(index)


def generate_plan(_client: SycamoreQueryClient, query: str, index: str) -> LogicalPlan:
    return _client.generate_plan(query, index, get_schema(_client, index))


def run_plan(_client: SycamoreQueryClient, plan: LogicalPlan) -> Tuple[str, Any]:
    return _client.run_plan(plan)


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


class QueryNodeTrace:
    def __init__(self, trace_dir: str, node_id: str):
        self.trace_dir = trace_dir
        self.node_id = node_id
        self.docs: List[Dict[str, Any]] = []
        self.readdata()

    def readdata(self):
        directory = os.path.join(self.trace_dir, self.node_id)
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                with open(f, "rb") as file:
                    try:
                        doc = pickle.load(file)
                    except EOFError:
                        continue
                    # For now, skip over MetadataDocuments.
                    if "metadata" in doc.keys():
                        continue

                    # Flatten properties.
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

                    self.docs.append(doc)

        # Group by the parent ID.
        self.parent_docs = [x for x in self.docs if x.get("parent_id") is None]
        print(self.parent_docs)
        for doc in self.parent_docs:
            doc["children"] = [x for x in self.docs if x.get("parent_id") == doc.get("doc_id")]

    def show(self):
        if not self.parent_docs:
            st.write(f"Result of node {self.node_id} — **no** documents")
            st.write("No data.")
            return
        st.write(f"Result of node {self.node_id} — **{len(self.parent_docs)}** documents")
        DEFAULT_COLUMNS = [
            "properties.path",
            "properties.entity.accidentNumber",
            "properties.entity.dateAndTime",
            "properties.entity.location",
            "properties.entity.aircraft",
            "properties.entity.registration",
            "properties.entity.injuries",
            "properties.entity.aircraftDamage",
        ]
        all_columns = self.parent_docs[0].keys()
        columns = DEFAULT_COLUMNS + [x for x in all_columns if x not in DEFAULT_COLUMNS]

        # Transpose data to a dict where each key is a column, and each value is a list of rows.
        data = {col: [] for col in columns}
        for row in self.parent_docs:
            for col in columns:
                data[col].append(row.get(col))

        df = pd.DataFrame(data)
        st.dataframe(
            df,
            column_order=columns,
            column_config={
                "properties.path": st.column_config.LinkColumn(
                    "Document link",
                    validate=r"^[a-z]+://[a-z\.\/]+$",
                    max_chars=100,
                    #            display_text=r"https://(.*?)\.streamlit\.app"
                )
            },
        )


class QueryTrace:
    def __init__(self, trace_dir: str):
        self.trace_dir = trace_dir
        self.node_traces = [QueryNodeTrace(trace_dir, node_id) for node_id in sorted(os.listdir(self.trace_dir))]

    def show(self):
        for node_trace in self.node_traces:
            node_trace.show()


@st.fragment
def show_query_traces(trace_dir: str, query_id: str):
    """Show the query traces in the given trace_dir."""
    trace_dir = os.path.join(trace_dir, query_id)
    QueryTrace(trace_dir).show()
