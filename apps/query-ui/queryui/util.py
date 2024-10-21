import base64
import json
import os
import pickle
from typing import Any, List, Set, Tuple

import boto3
import requests
import pandas as pd
import streamlit as st

from sycamore.docset import DocSet
from sycamore.data import MetadataDocument
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.planner import PlannerExample
from sycamore.query.schema import OpenSearchSchema
from sycamore.query.result import SycamoreQueryResult

from queryui.configuration import get_sycamore_query_client


def get_schema(_client: SycamoreQueryClient, index: str) -> OpenSearchSchema:
    """Return the OpenSearch schema for the given index."""
    return _client.get_opensearch_schema(index)


def generate_plan(_client: SycamoreQueryClient, query: str, index: str, examples: List[PlannerExample]) -> LogicalPlan:
    """Generate a query plan for the given query and index."""
    return _client.generate_plan(query, index, get_schema(_client, index), examples=examples)


def run_plan(_client: SycamoreQueryClient, plan: LogicalPlan) -> SycamoreQueryResult:
    """Run the given plan."""
    return _client.run_plan(plan)


def get_opensearch_indices() -> Set[str]:
    """Return a list of OpenSearch indices."""
    return {x for x in get_sycamore_query_client().get_opensearch_indices() if not x.startswith(".")}


def result_to_string(result: Any) -> str:
    """Convert the given query result to a string."""
    if isinstance(result, str):
        # We got a straight string response from the query plan, which means we can
        # return it directly.
        return result
    elif isinstance(result, DocSet):
        # We got a DocSet.
        return docset_to_string(result)
    else:
        # Fall back to string representation.
        return str(result)


NUM_DOCS_GENERATE = 60
NUM_TEXT_CHARS_GENERATE = 2500


def docset_to_string(docset: DocSet, html: bool = True) -> str:
    """Render the given DocSet as a string.

    Args:
        docset: The DocSet to render.
        html: Whether to render as HTML. Otherwise, JSON is used.
    """

    BASE_PROPS = [
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
    retval = ""
    for doc in docset.take(NUM_DOCS_GENERATE):
        if isinstance(doc, MetadataDocument):
            continue
        if html:
            retval += f"**{doc.properties.get('path')}** \n"

            retval += "| Property | Value |\n"
            retval += "|----------|-------|\n"

            props_dict = doc.properties.get("entity", {})
            props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})

            for k, v in props_dict.items():
                retval += f"| {k} | {v} |\n"

            retval += "\n\n"
            text_content = (
                doc.text_representation[:NUM_TEXT_CHARS_GENERATE] if doc.text_representation is not None else None
            )
            if text_content:
                retval += f'*..."{text_content}"...* <br><br>'
        else:
            props_dict = doc.properties.get("entity", {})
            props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
            props_dict["text_representation"] = (
                doc.text_representation[:NUM_TEXT_CHARS_GENERATE] if doc.text_representation is not None else None
            )
            # The DocumentSource is not JSON serializable.
            if "_doc_source" in props_dict:
                props_dict["_doc_source"] = None

            retval += json.dumps(props_dict, indent=2) + "\n"
    return retval


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
    """Display a preview of the given document in a dialog."""
    PDFPreview(path).show()


class QueryNodeTrace:
    """Helper class to read and display the trace of a single query node."""

    # The order here is chosen to ensure that the most important columns are shown first.
    # The logic below ensure that additional columns are also shown, and that empty columns
    # are omitted.
    COLUMNS = [
        "properties.path",
        "properties.key",
        "properties.count",
        "doc_id",
        "parent_id",
        "text_representation",
    ]

    def __init__(self, result: SycamoreQueryResult, node_id: int):
        self.result = result
        self.node_id = node_id
        self.df = None
        self.num_files = 0
        self.readdata()

    def readfile(self, f):
        """Read the given trace file."""
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
                """Format the given value for display."""
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
        """Read the trace data for this node."""

        if self.node_id not in self.result.trace_dirs:
            return

        directory = self.result.trace_dirs[self.node_id]
        docs = []

        # We need to read all of the individual docs to ensure we get all of the parent
        # docs. With a very large number of docs, we are likely to blow out memory doing
        # this. Unfortunately there's no easy way to tell up-front that a given stage in
        # the pipeline has a mix of parent and child docs, unless we do two passes.
        # Just a heads up that with a larger number of docs, we may need to revisit this.
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                self.num_files += 1
                newdoc = self.readfile(f)
                if newdoc:
                    docs.append(newdoc)

        # Only keep parent docs if there are child docs in the list.
        parent_docs = [x for x in docs if x.get("parent_id") is None]
        if len(parent_docs) > 0:
            docs = parent_docs

        # Transpose data to a dict where each key is a column, and each value is a list of rows.
        if len(docs) > 0:
            all_keys = {k for d in docs for k in d.keys()}
            data = {col: [] for col in all_keys}
            for row in docs:
                for col in all_keys:
                    data[col].append(row.get(col))
            self.df = pd.DataFrame(data)

    def show(self, node):
        """Render the trace data."""
        st.subheader(f"Node {self.node_id} - {node.node_type if node else 'n/a'}")
        st.markdown(f"*Description: {node.description if node else 'n/a'}*")
        if self.df is None or not len(self.df):
            if self.num_files > 0:
                st.write(":green[Cache hit - node execution skipped!]")
            else:
                st.write(f":green[Node type {node.node_type} does not produce document traces]")
            return

        all_columns = list(self.df.columns)
        column_order = [c for c in self.COLUMNS if c in all_columns]
        column_order += [c for c in all_columns if c not in column_order]
        st.write(f"**{len(self.df)} documents**")
        st.dataframe(self.df, column_order=column_order)


class QueryTrace:
    """Helper class used to read and display query traces."""

    def __init__(self, result: SycamoreQueryResult):
        self.node_traces = []
        self.query_plan = result.plan
        for node_id in sorted(result.plan.nodes.keys()):
            self.node_traces += [QueryNodeTrace(result, node_id)]

    def show(self):
        tab1, tab2 = st.tabs(["Node data", "Query plan"])
        with tab1:
            for node_trace in self.node_traces:
                node_trace.show(self.query_plan.nodes.get(int(node_trace.node_id), None))
        with tab2:
            if self.query_plan is not None:
                st.write(f"Query: {self.query_plan.query}")
                st.write(self.query_plan)
            else:
                st.write("No query plan found")


@st.fragment
def show_query_traces(result: SycamoreQueryResult):
    """Show the query traces for the given query result."""
    QueryTrace(result).show()
