import io
import os
import pickle
import zipfile
import pandas as pd

import streamlit as st

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
    ]
)

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
                    
                    demo_list = {}
                    if "properties" in doc:
                        for property in doc["properties"]:
                            if property in BASE_PROPS:
                                continue
                            if isinstance(doc["properties"][property], dict):
                                for nested_property in doc["properties"][property]:
                                    demo_list[".".join(["properties", property, nested_property])] = doc["properties"][
                                        property
                                    ][nested_property]
                            else:
                                demo_list[".".join(["properties", property])] = doc["properties"][property]
                    demo_list["text_representation"] = doc["text_representation"]
                    data_list.append(demo_list)

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
