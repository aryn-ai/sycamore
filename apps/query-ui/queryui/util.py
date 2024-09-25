import logging
import os
import pickle
import pandas as pd
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel
import ray
import streamlit as st
from sycamore.docset import DocSet
from sycamore.data import MetadataDocument
from sycamore.executor import _ray_logging_setup
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.planner import QueryPlan, PlannerExample
from sycamore.query.schema import OpenSearchSchema
from yaml import safe_load


from configuration import get_sycamore_query_client


class LunaQueryExample(BaseModel):
    """Represents a single example query in the Luna configuration file."""

    query: str
    plan: QueryPlan


class LunaIndexConfig(BaseModel):
    """Represents the configuration for a single index in the Luna configuration file."""

    data_schema: Optional[OpenSearchSchema]
    examples: Optional[List[LunaQueryExample]]

    def get_planner_examples(self):
        """Return a list of PlannerExample objects for this index."""
        return [
            PlannerExample(query=example.query, data_schema=self.data_schema, plan=example.plan)
            for example in self.examples
        ]


class LunaConfig(BaseModel):
    """The format of the Luna configuration file."""

    indices: Dict[str, LunaIndexConfig]


def read_config_file(config_file: str) -> LunaConfig:
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return LunaConfig(**safe_load(f))
    except FileNotFoundError:
        return LunaConfig(indices={})


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
    """Get the schema for the given index."""
    return _client.get_opensearch_schema(index)


def generate_plan(_client: SycamoreQueryClient, query: str, index: str, examples: Optional[Any] = None) -> LogicalPlan:
    """Generate a query plan for the given query and index."""
    return _client.generate_plan(query, index, get_schema(_client, index), examples=examples)


def run_plan(_client: SycamoreQueryClient, plan: LogicalPlan) -> Tuple[str, Any]:
    """Run the given query plan and return the result."""
    return _client.run_plan(plan)


def get_opensearch_indices() -> Set[str]:
    """Get the OpenSearch indices available in the system."""
    return {x for x in get_sycamore_query_client().get_opensearch_incides() if not x.startswith(".")}


def result_to_string(result: Any) -> str:
    """Convert a query result to a string representation."""

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


def docset_to_string(docset: DocSet) -> str:
    """Convert a DocSet to a string representation."""

    NUM_DOCS_GENERATE = 60
    NUM_TEXT_CHARS_GENERATE = 2500
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
        retval += f"**{doc.properties.get('path')}** page: {doc.properties.get('page_number', 'meta')}  \n"

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
        retval += f'*..."{text_content}"...* <br><br>'
    return retval


class QueryNodeTrace:
    """Display query traces for a single query node."""

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
        "properties.entity.conditions",
        "properties.entity.windSpeed",
        "properties.entity.visibility",
        "properties.entity.lowestCeiling",
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
    """Display query traces from the given trace_dir."""

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
