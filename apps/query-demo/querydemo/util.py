import base64
import logging
import os
import pickle
import pandas as pd
from typing import Any, Dict, Set, Tuple

import boto3
import ray
import requests
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


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse an S3 path into a bucket and key."""
    s3_path = s3_path.replace("s3://", "")
    bucket, key = s3_path.split("/", 1)
    return bucket, key


class PDFPreview:
    """Display a preview of the given PDF file."""

    def __init__(self, path: str):
        self.path = path

    def show(self):
        if self.path.startswith("s3://"):
            bucket, key = parse_s3_path(self.path)
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
        elif self.path.startswith("http"):
            content = requests.get(self.path, timeout=30).content
        else:
            st.write(f"Unknown path format: {self.path}")
            return

        st.text(self.path)
        encoded = base64.b64encode(content).decode("utf-8")
        pdf_display = (
            f'<iframe src="data:application/pdf;base64,{encoded}" '
            + 'width="600" height="800" type="application/pdf"></iframe>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)


@st.dialog("Document preview", width="large")
def show_pdf_preview(path: str):
    PDFPreview(path).show()


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
    # The order here is chosen to ensure that the most important columns are shown first.
    # The logic below ensure that additional columns are also shown, and that empty columns
    # are omitted.
    COLUMNS = [
        "properties.path",
        "properties.entity.accidentNumber",
        "properties.key",
        "properties.count",
        "properties.entity.location",
        "properties.entity.dateAndTime",
        "properties.entity.aircraft",
        "properties.entity.registration",
        "properties.entity.injuries",
        "properties.entity.aircraftDamage",
        "text_representation",
        "doc_id",
        "parent_id",
    ]

    def __init__(self, trace_dir: str, node_id: str):
        self.trace_dir = trace_dir
        self.node_id = node_id
        self.df = None
        self.readdata()

    def readfile(self, f):
        with open(f, "rb") as file:
            try:
                doc = pickle.load(file)
            except EOFError:
                return None

            # For now, skip over MetadataDocuments.
            if "metadata" in doc.keys():
                return None

            # Flatten properties.
            if "properties" in doc:
                for prop in doc["properties"]:
                    if isinstance(doc["properties"][prop], dict):
                        for nested_property in doc["properties"][prop]:
                            doc[".".join(["properties", prop, nested_property])] = doc["properties"][prop][
                                nested_property
                            ]
                    else:
                        doc[".".join(["properties", prop])] = doc["properties"][prop]
                doc.pop("properties")

            # Keep only the columns we care about.
            # Limit size of each to avoid blowing out memory.
            def format_value(value):
                MAX_CHARS = 1024
                if isinstance(value, str):
                    if len(value) > MAX_CHARS:
                        return value[:MAX_CHARS] + "..."
                    else:
                        return value
                if isinstance(value, dict):
                    return format_value(str(value))
                if isinstance(value, list):
                    return format_value(str(value))
                return value

            row = {k: format_value(doc.get(k)) for k in doc.keys() if k in self.COLUMNS or k.startswith("properties.")}
            return row

    def readdata(self):
        directory = os.path.join(self.trace_dir, self.node_id)
        docs = []

        # We need to read all of the individual docs to ensure we get all of the parent
        # docs. With a very large number of docs, we are likely to blow out memory doing
        # this. Unfortunately there's no easy way to tell up-front that a given stage in
        # the pipeline has a mix of parent and child docs, unless we do two passes.
        # Just a heads up that with a larger number of docs, we may need to revisit this.
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                newdoc = self.readfile(f)
                if newdoc:
                    docs.append(newdoc)

        # Only keep parent docs if there are child docs in the list.
        parent_docs = [x for x in docs if x.get("parent_id") is None]
        if parent_docs:
            docs = parent_docs

        # Transpose data to a dict where each key is a column, and each value is a list of rows.
        if docs:
            all_keys = {k for d in docs for k in d.keys()}
            data = {col: [] for col in all_keys}
            for row in docs:
                for col in all_keys:
                    data[col].append(row.get(col))
            self.df = pd.DataFrame(data)

    def show(self):
        if self.df is None or not len(self.df):
            st.write(f"Result of node {self.node_id} — **no** documents")
            st.write("No data.")
            return

        all_columns = list(self.df.columns)
        column_order = [c for c in self.COLUMNS if c in all_columns]
        column_order += [c for c in all_columns if c not in column_order]
        st.write(f"Result of node {self.node_id} — **{len(self.df)}** documents")
        st.dataframe(self.df, column_order=column_order)


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
