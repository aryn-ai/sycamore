import datetime
from html.parser import HTMLParser
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import marko
from marko.md_renderer import MarkdownRenderer
import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
import streamlit as st
import sycamore
from sycamore.data import OpenSearchQuery
from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.query.client import SycamoreQueryClient
from util import generate_plan, run_plan, show_query_traces, show_pdf_preview, ray_init

NUM_DOCS_GENERATE = 60
NUM_DOCS_PREVIEW = 10
NUM_TEXT_CHARS_GENERATE = 2500

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OPENSEARCH_INDEX = "const_ntsb"
OS_CONFIG = {"search_pipeline": "hybrid_pipeline"}
OS_CLIENT_ARGS = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "http_compress": True,
    "http_auth": ("admin", "admin"),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120,
}
S3_CACHE_PATH = "s3://aryn-temp/llm_cache/luna/query-demo"

EXAMPLE_QUERIES = [
    "How many incidents were there in Washington in 2023?",
    "What was the breakdown of aircraft types for incidents with substantial damage?",
    "Show me incidents involving Piper aircraft",
    "Show the details on accident ERA23LA153",
]

WELCOME_MESSAGE = f"""Welcome to the NTSB incident query demo! You can ask me questions about
[NTSB incident reports](https://carol.ntsb.gov/), and I'll do my best to answer them. Feel free
to ask about specific incidents,o aggregate statistics, or anything else you're curious about.
If you're not sure what to ask, you can try one of the following example queries:

{"".join([f"<SuggestedQuery query='{query}' />" for query in EXAMPLE_QUERIES])}
"""

SYSTEM_PROMPT = """You are a helpful agent that answers questions about NTSB
(National Transportation Safety Board) incidents. You have access to a database of incident
reports, each of which has an associated PDF document, as well as metadata about the incident
including the location, date, aircraft type, and more. You can answer questions about the
contents of individual reports, as well as aggregate statistics about the incidents in the
database. You can perform actions such as filtering, sorting, and aggregating the data to
answer questions. You can also provide links to relevant documents and data sources.

All your responses should be in MDX, which is Markdown augmented with JSX components. 
As an example, your response could include a table of data, a link to a document, or a
component such as a button. Below is an example of MDX output that you might generate:

```Here are the latest incidents referring to bad weather:

<Table>
    <TableHeader>
        <TableCell>Incident ID</TableCell>
        <TableCell>Date</TableCell>
        <TableCell>Location</TableCell>
        <TableCell>Aircraft Type</TableCell>
    </TableHeader>
    <TableRow>
        <TableCell>ERA23LA153</TableCell>
        <TableCell>2023-01-15</TableCell>
        <TableCell>Seattle, WA</TableCell>
        <TableCell>Cessna 172</TableCell>
    </TableRow>
</Table>
```

Additional markdown text can follow the use of a JSX component, and JSX components can be
inlined in the markdown text as follows:

```For more information about this incident, please see the following document:
  <Preview path="s3://aryn-public/samples/sampledata1.pdf" />.
```

Do not include a starting ``` and closing ``` line in your reply. Just respond with the MDX itself.
Do not include extra whitespace that is not needed for the markdown interpretation. For instance,
if a JSX component has a property that's a JSON object, encode it into a single line, like so:

<Component prop="{\"key1\": \"value1\", \"key2": \"value2\"}" />

The following JSX components are available for your use:
  * <Table>
      <TableHeader>
        <TableCell>Column name 1</TableCell>
        <TableCell>Column name 2</TableCell>
      </TableHeader>
      <TableRow>
        <TableCell>Value 1</TableCell>
        <TableCell>Value 2</TableCell>
      </TableRow>
      <TableRow>
        <TableCell>Value 3</TableCell>
        <TableCell>Value 4</TableCell>
      </TableRow>
    </Table>
    Displays a table showing the provided data. 

  * <Preview path="s3://aryn-public/samples/sampledata1.pdf" />
    Displays an inline preview of the provided document. You may provide an S3 path or a URL.
    ALWAYS use a <Preview> instead of a regular link whenever a document is mentioned.

  * <SuggestedQuery query="How many incidents were there in Washington in 2023?" />
    Displays a button showing a query that the user might wish to consider asking next.

  * <Map>
     <MapMarker lat="47.6062" lon="-122.3321" />
     <MapMarker lat="47.7105" lon="-122.4406" />
    </Map>
    Displays a map with markers at the provided coordinates.

Multiple JSX components can be used in your reply. You may ONLY use these specific JSX components
in your responses. Other than these components, you may ONLY use standard Markdown syntax.

Please use <Preview> any time a specific document or incident is mentioned, and <Map> any time
there is an opportunity to refer to a location.

Please suggest 1-3 follow-on queries (using the <SuggestedQuery> component) that the user might
ask, based on the response to the user's question.
"""

PLANNER_EXAMPLE_SCHEMA = """
DATA_SCHEMA: {
  "text_representation": "<class 'str'> (e.g., Can be assumed to have all other details)",
  "properties.entity.dateTime": "<class 'str'> (e.g., 2023-01-12T11:00:00, 2023-01-11T18:09:00,
    2023-01-10T16:43:00, 2023-01-28T19:02:00, 2023-01-12T13:00:00)",
  "properties.entity.dateAndTime": "<class 'str'> (e.g., January 28, 2023 19:02:00, January 10, 2023
    16:43:00, January 11, 2023 18:09:00, January 12, 2023 13:00:00, January 12, 2023 11:00:00)",
  "properties.entity.lowestCeiling": "<class 'str'> (e.g., Broken 3800 ft AGL, Broken 6500 ft AGL,
    Overcast 500 ft AGL, Overcast 1800 ft AGL)",
  "properties.entity.aircraftDamage": "<class 'str'> (e.g., Substantial, None, Destroyed)",
  "properties.entity.conditions": "<class 'str'> (e.g., , Instrument (IMC), IMC, VMC, Visual (VMC))",
  "properties.entity.departureAirport": "<class 'str'> (e.g., Somerville, Tennessee, Colorado Springs,
    Colorado (FLY), Yelm; Washington, Winchester, Virginia (OKV), San Diego, California (KMYF))",
  "properties.entity.accidentNumber": "<class 'str'> (e.g., CEN23FA095, ERA2BLAT1I, WPR23LA088,
    ERA23FA108, WPR23LA089)",
  "properties.entity.windSpeed": "<class 'str'> (e.g., , 10 knots, 7 knots, knots, 19 knots gusting
    to 22 knots)",
  "properties.entity.day": "<class 'str'> (e.g., 2023-01-12, 2023-01-10, 2023-01-20, 2023-01-11,
    2023-01-28)",
  "properties.entity.destinationAirport": "<class 'str'> (e.g., Somerville, Tennessee, Yelm;
    Washington, Agua Caliente Springs, California, Liberal, Kansas (LBL), Alabaster, Alabama (EET))",
  "properties.entity.location": "<class 'str'> (e.g., Hooker, Oklahoma, Somerville, Tennessee, Yelm;
    Washington, Agua Caliente Springs, California, Dayton, Virginia)",
  "properties.entity.operator": "<class 'str'> (e.g., On file, First Team Pilot Training LLC,
    file On, Anderson Aviation LLC, Flying W Ranch)",
  "properties.entity.temperature": "<class 'str'> (e.g., 18'C /-2'C, 15.8C, 13'C, 2C / -3C)",
  "properties.entity.registration": "<class 'str'> (e.g., N5841W, N2875K, N6482B, N43156, N225V)",
  "properties.entity.visibility": "<class 'str'> (e.g., , miles, 0.5 miles, 7 miles, 10 miles)",
  "properties.entity.aircraft": "<class 'str'> (e.g., Piper PA-32R-301, Beech 95-C55, Cessna 172,
    Piper PA-28-160, Cessna 180K)",
  "properties.entity.conditionOfLight": "<class 'str'> (e.g., , Night/dark, Night, Day, Dusk)",
  "properties.entity.windDirection": "<class 'str'> (e.g., , 190\\u00b0, 200, 2005, 040\\u00b0)",
  "properties.entity.lowestCloudCondition": "<class 'str'> (e.g., , Broken 3800 ft AGL, Overcast
    500 ft AGL, Clear, Overcast 200 ft AGL)",
  "properties.entity.injuries": "<class 'str'> (e.g., Minor, Fatal, None, 3 None, 2 None)",
  "properties.entity.flightConductedUnder": "<class 'str'> (e.g.,
  Part 91: General aviation Instructional, Part 135: Air taxi & commuter Non-scheduled, Part 91:
    General aviation Personal, Part 135: Air taxi & commuter Scheduled, Part 91: General aviation
    Business)"
}
  """

PLANNER_EXAMPLES = [
    (
        "List the incidents in Georgia in 2023.",
        [
            {
                "operatorName": "QueryDatabase",
                "description": "Get all the incident reports",
                "index": OPENSEARCH_INDEX,
                "node_id": 0,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "properties.entity.dateTime": {
                                        "gte": "2023-01-01T00:00:00",
                                        "lte": "2023-12-31T23:59:59",
                                        "format": "strict_date_optional_time",
                                    }
                                }
                            },
                            {"match": {"properties.entity.location": "Georgia"}},
                        ]
                    }
                },
            },
        ],
    ),
    (
        "Show the incidents involving Piper aircraft.",
        [
            {
                "operatorName": "QueryDatabase",
                "description": "Get all the incident reports",
                "index": OPENSEARCH_INDEX,
                "node_id": 0,
                "query": {"match": {"properties.entity.aircraft": "Piper"}},
            },
        ],
    ),
    (
        "How many incidents happened in clear weather?",
        [
            {
                "operatorName": "QueryDatabase",
                "description": "Get all the incident reports in clear weather",
                "index": OPENSEARCH_INDEX,
                "node_id": 0,
                "query": {"match": {"properties.entity.conditions": "VMC"}},
            },
            {
                "operatorName": "Count",
                "description": "Count the number of incidents",
                "distinct_field": "properties.entity.accidentNumber",
                "input": [0],
                "node_id": 1,
            },
        ],
    ),
    (
        "What types of aircrafts were involved in accidents in California?",
        [
            {
                "operatorName": "QueryDatabase",
                "description": "Get all the incident reports in California",
                "index": OPENSEARCH_INDEX,
                "query": {"match": {"properties.entity.location": "California"}},
                "node_id": 0,
            },
            {
                "operatorName": "TopK",
                "description": "Get the types of aircraft involved in incidents in California",
                "field": "properties.entity.aircraft",
                "primary_field": "properties.entity.accidentNumber",
                "K": 100,
                "descending": False,
                "llm_cluster": False,
                "llm_cluster_instruction": None,
                "input": [0],
                "node_id": 1,
            },
        ],
    ),
    (
        "Which aircraft accidents in California in 2023 occurred when the wind was stronger than 4 knots?",
        [
            {
                "operatorName": "QueryDatabase",
                "description": "Get all the incident reports in California in 2023",
                "index": OPENSEARCH_INDEX,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "properties.entity.dateTime": {
                                        "gte": "2023-01-01T00:00:00",
                                        "lte": "2023-12-31T23:59:59",
                                        "format": "strict_date_optional_time",
                                    }
                                }
                            },
                            {"match": {"properties.entity.location": "California"}},
                        ]
                    }
                },
                "node_id": 0,
            },
            {
                "operatorName": "LlmFilter",
                "description": "Filter to reports with wind speed greater than 4 knots",
                "index": OPENSEARCH_INDEX,
                "question": "Is the wind speed greater than 4 knots?",
                "field": "properties.entity.windSpeed",
                "input": [0],
                "node_id": 1,
            },
        ],
    ),
    (
        "Which three aircraft types were involved in the most accidents?",
        [
            {
                "operatorName": "QueryDatabase",
                "description": "Get all the incident reports",
                "index": OPENSEARCH_INDEX,
                "node_id": 0,
                "query": {"match_all": {}},
            },
            {
                "operatorName": "TopK",
                "description": "Get the top three aircraft types involved in accidents",
                "field": "properties.entity.aircraft",
                "primary_field": "properties.entity.accidentNumber",
                "K": 3,
                "descending": True,
                "llm_cluster": False,
                "llm_cluster_instruction": None,
                "input": [0],
                "node_id": 1,
            },
        ],
    ),
    (
        "Show some incidents where pilot training was mentioned as a cause",
        [
            {
                "operatorName": "QueryVectorDatabase",
                "description": "Get incident reports mentioning pilot training",
                "index": OPENSEARCH_INDEX,
                "query_phrase": "pilot training",
                "node_id": 0,
            },
        ],
    ),
    (
        "Show all incidents involving a Cessna 172 aircraft",
        [
            {
                "operatorName": "QueryDatabase",
                "description": "Get all the incident reports involving a Cessna 172 aircraft",
                "index": OPENSEARCH_INDEX,
                "query": {"match": {"properties.entity.aircraft": "Cessna 172"}},
                "node_id": 0,
            },
        ],
    ),
]


def generate_example_prompt() -> str:
    prompt = f"The following examples refer to the following data schema:\n{PLANNER_EXAMPLE_SCHEMA}\n\n"
    for index, (query, plan) in enumerate(PLANNER_EXAMPLES):
        plan_string = json.dumps(plan)
        prompt += f"Example {index+1}:\nUSER QUESTION: {query}\nAnswer:\n{plan_string}\n\n"
    return prompt


class MDXParser(HTMLParser):

    def __init__(self, chat_message: "ChatMessage"):
        super().__init__()
        self.chat_message = chat_message
        self.in_table = False
        self.in_header = False
        self.in_header_cell = False
        self.in_row = False
        self.in_cell = False
        self.table_data: List[List[Any]] = []
        self.row_data: List[Any] = []
        self.header_data: List[Any] = []
        self.in_map = False
        self.map_data: List[Tuple[float, float]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.table_data = []
        elif tag == "tableheader" and self.in_table:
            self.in_header = True
            self.header_data = []
        elif tag == "tablerow" and self.in_table:
            self.in_row = True
            self.row_data = []
        elif tag == "tablecell":
            if self.in_header:
                self.in_header_cell = True
            elif self.in_row:
                self.in_cell = True
        elif tag == "map":
            self.in_map = True
            self.map_data = []
        elif tag == "mapmarker" and self.in_map:
            lat = float(dict(attrs).get("lat"))
            lon = float(dict(attrs).get("lon"))
            self.map_data.append((lat, lon))

    def handle_endtag(self, tag):
        if tag == "tablecell":
            if self.in_header_cell:
                self.in_header_cell = False
            elif self.in_cell:
                self.in_cell = False
        elif tag == "tableheader" and self.in_header:
            self.in_header = False
        elif tag == "tablerow" and self.in_row:
            self.in_row = False
            self.table_data.append(self.row_data)
        elif tag == "table" and self.in_table:
            self.in_table = False
            self.chat_message.render_table(self.header_data, self.table_data)
        elif tag == "map" and self.in_map:
            self.in_map = False
            self.chat_message.render_map(self.map_data)

    def handle_startendtag(self, tag, attrs):
        if tag == "preview":
            path = dict(attrs).get("path")
            if path:
                self.chat_message.render_preview(path)
            else:
                self.chat_message.render_markdown("No path provided for preview.")
        elif tag == "suggestedquery":
            query = dict(attrs).get("query")
            if query:
                self.chat_message.render_suggested_query(query)
            else:
                self.chat_message.render_markdown("No query provided for suggested query")
        elif tag == "mapmarker" and self.in_map:
            lat = float(dict(attrs).get("lat"))
            lon = float(dict(attrs).get("lon"))
            self.map_data.append((lat, lon))

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        if self.in_header_cell:
            self.header_data.append(data)
        elif self.in_cell:
            self.row_data.append(data)
        else:
            self.chat_message.render_markdown(data)
            # Scan for previewable markdown links and add previews.
            markdown_link_regexp = r"!?\[([^\]]+)\]\(([^\)]+)\)"
            for m in re.finditer(markdown_link_regexp, data):
                if m and m.group(2).startswith("s3://"):
                    self.chat_message.render_preview(m.group(2))
                    data = re.sub(markdown_link_regexp, "\1", data)


class JSXElementParser(HTMLParser):
    tag = None
    props = None

    def handle_startendtag(self, tag, attrs):
        self.tag = tag
        self.props = dict(attrs)


class MDRenderer(MarkdownRenderer):
    """A Marko renderer that replaces S3 links with links to the demo instance."""

    def __init__(self):
        super().__init__()

    def render_link(self, element: marko.inline.Link) -> str:
        if element.dest.startswith("s3://"):
            # Replace S3 links with a link to the demo instance.
            # (This is somewhat of a temporary hack until we build out better UI for this.)
            element.dest = element.dest.replace("s3://", "https://luna-demo.dev.aryn.ai/doc/")
        return super().render_link(element)


class ChatMessageExtra:
    def __init__(self, name: str, content: Optional[Any] = None):
        self.name = name
        self.content = content

    def show(self):
        st.write(self.content)


class ChatMessageTraces(ChatMessageExtra):
    def __init__(self, name: str, query_id: str):
        super().__init__(name)
        self.query_id = query_id

    def show(self):
        show_query_traces(st.session_state.trace_dir, self.query_id)


class ChatMessage:
    def __init__(
        self,
        message: Optional[Dict[str, Any]] = None,
        before_extras: Optional[List[ChatMessageExtra]] = None,
        after_extras: Optional[List[ChatMessageExtra]] = None,
    ):
        self.message_id = st.session_state.next_message_id
        st.session_state.next_message_id += 1
        self.timestamp = datetime.datetime.now()
        self.message = message or {}
        self.before_extras = before_extras or []
        self.after_extras = after_extras or []
        self.widget_key = 0

    def show(self):
        self.widget_key = 0
        with st.chat_message(self.message.get("role", "assistant")):
            self.show_content()

    def show_content(self):
        for extra in self.before_extras:
            with st.expander(extra.name):
                extra.show()
        if self.message.get("content"):
            content = self.message.get("content")
            self.render_markdown_with_jsx(content)
        for extra in self.after_extras:
            with st.expander(extra.name):
                extra.show()

    def button(self, label: str, **kwargs):
        return st.button(label, key=self.next_key(), **kwargs)

    def download_button(self, label: str, content: bytes, filename: str, file_type: str, **kwargs):
        return st.download_button(label, content, filename, file_type, key=self.next_key(), **kwargs)

    def next_key(self) -> str:
        key = f"{self.message_id}-{self.widget_key}"
        self.widget_key += 1
        return key

    def render_markdown_with_jsx(self, text: str):
        print(f"Rendering MDX:\n{text}\n")
        mdxparser = MDXParser(self)
        mdxparser.feed(text)

    def render_markdown(self, text: str):
        st.markdown(text)

    def render_table(self, columns: List[str], rows: List[List[str]]):
        if not rows:
            return
        num_cols = max([len(row) for row in rows])
        if not columns:
            columns = [f"Column {i}" for i in range(1, num_cols + 1)]
        if len(columns) < num_cols:
            columns += [f"Column {i}" for i in range(len(columns) + 1, num_cols + 1)]
        if len(columns) > num_cols:
            columns = columns[:num_cols]

        data: Dict[str, List[Any]] = {col: [] for col in columns}
        for row in rows:
            for index, col in enumerate(columns):
                try:
                    data[col].append(row[index])
                except IndexError:
                    # This can happen if we end up with missing cells in the MDX.
                    data[col].append(None)

        df = pd.DataFrame(data)
        st.dataframe(df)

    def render_preview(self, path: str):
        with st.container(border=True):
            col1, col2 = st.columns([0.75, 0.25])
            with col1:
                st.text("📄 " + path)
            with col2:
                if st.button("View document", key=self.next_key()):
                    show_pdf_preview(path)

    def render_suggested_query(self, query: str):
        if self.button(query):
            st.session_state.user_query = query

    def render_map(self, data: List[Tuple[float, float]]):
        df = pd.DataFrame(data, columns=["lat", "lon"])
        st.map(df, color="#00ff00", size=100)

    def to_dict(self) -> Optional[Dict[str, Any]]:
        return self.message

    def __str__(self):
        return f"{self.message.get('role', 'assistant')}: {self.message.get('content')}"


TOOLS: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "queryDataSource",
            "description": """Run a query against the backend data source. The query should be a natural language 
            query and can represent operations such as filters, aggregations, sorting, formatting,
            and more. This function should only be called when new data is needed; if the data
            is already available in the message history, it should be used directly.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


TOOLS_RAG: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "queryDataSource",
            "description": """Run a query against the backend data source. The query should be a
            natural language query. This function should only be called when new data is needed;
            if the data is already available in the message history, it should be used directly.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def get_embedding_model_id() -> str:
    """Get the embedding model ID from the OpenSearch instance."""
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {"name": "all-MiniLM-L6-v2"},
                    },
                    {
                        "term": {"model_config.model_type": "bert"},
                    },
                ],
            },
        },
    }
    with requests.get(
        "https://localhost:9200/_plugins/_ml/models/_search", json=query, verify=False, timeout=60
    ) as resp:
        res = json.loads(resp.text)
        return res["hits"]["hits"][0]["_id"]


def do_rag_query(query: str, index: str) -> str:
    """Uses OpenSearch to perform a RAG query."""
    embedding_model_id = get_embedding_model_id()
    search_pipeline = "hybrid_rag_pipeline"
    llm = "gpt-4o"
    rag_query = OpenSearchQuery()
    rag_query["index"] = index
    rag_query["query"] = {
        "_source": {"excludes": ["embedding"]},
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"text_representation": query}},
                    {
                        "neural": {
                            "embedding": {
                                "query_text": query,
                                "model_id": embedding_model_id,
                                "k": 100,
                            }
                        }
                    },
                ]
            }
        },
        "size": 20,
    }

    # RAG params
    rag_query["params"] = {"search_pipeline": search_pipeline}
    rag_query["query"]["ext"] = {
        "generative_qa_parameters": {
            "llm_question": query,
            "context_size": 10,
            "llm_model": llm,
        }
    }

    with st.expander("RAG query"):
        st.write(rag_query)

    with st.spinner("Running RAG query..."):
        osq = OpenSearchQueryExecutor(OS_CLIENT_ARGS)
        rag_result = osq.query(rag_query)["result"]
    return rag_result


def query_data_source(query: str, index: str) -> Tuple[Any, Optional[Any], Optional[str]]:
    """Run a Sycamore or RAG query.

    Returns a tuple of (query_result, query_plan, query_id).
    """

    if st.session_state.rag_only:
        return do_rag_query(query, index), None, None
    else:
        sqclient = SycamoreQueryClient(
            s3_cache_path=S3_CACHE_PATH,
            trace_dir=st.session_state.trace_dir,
        )
        with st.spinner("Generating plan..."):
            plan = generate_plan(sqclient, query, index, examples=generate_example_prompt())
            print(f"Generated plan:\n{plan}\n")
            # No need to show the prompt used in the demo.
            plan.llm_prompt = None
            with st.expander("Query plan"):
                st.write(plan)
        with st.spinner("Running Sycamore query..."):
            print("Running plan...")
            query_id, result = run_plan(sqclient, plan)
            print(f"Ran query ID: {query_id}")
        return result, plan, query_id


def docset_to_string(docset: sycamore.docset.DocSet) -> str:
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
        if isinstance(doc, sycamore.data.MetadataDocument):
            continue
        props_dict = doc.properties.get("entity", {})
        props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
        props_dict["text_representation"] = (
            doc.text_representation[:NUM_TEXT_CHARS_GENERATE] if doc.text_representation is not None else None
        )
        retval += json.dumps(props_dict, indent=2) + "\n"
    return retval


def show_messages():
    for msg in st.session_state.messages:
        if msg.message.get("role") not in ["user", "assistant"]:
            continue
        if msg.message.get("content") is None:
            continue
        msg.show()
    _, col2 = st.columns([0.75, 0.25])
    with col2:
        st.link_button("Send feedback", "/feedback", type="primary")



def do_query():
    prompt = st.session_state.user_query
    st.session_state.user_query = None
    print(f"User query: {prompt}")
    user_message = ChatMessage({"role": "user", "content": prompt})
    st.session_state.messages.append(user_message)
    user_message.show()

    assistant_message = None
    query_plan = None
    query_id = None
    tool_response_str = None
    with st.chat_message("assistant"):
        # We loop here because tool calls require re-invoking the LLM.
        while True:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [m.to_dict() for m in st.session_state.messages]
            with st.spinner("Running LLM query..."):
                response = openai_client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages,
                    tools=TOOLS_RAG if st.session_state.rag_only else TOOLS,
                    tool_choice="auto",
                )
            response_dict = response.choices[0].message.to_dict()
            assistant_message = ChatMessage(response_dict)
            st.session_state.messages.append(assistant_message)

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                tool_call_id = tool_calls[0].id
                tool_function_name = tool_calls[0].function.name
                tool_response_str = ""
                query_plan = None
                query_id = None
                # Try to catch any errors that might corrupt the message history here.
                try:
                    tool_args = json.loads(tool_calls[0].function.arguments)
                    if tool_function_name == "queryDataSource":
                        tool_query = tool_args["query"]
                        tool_response, query_plan, query_id = query_data_source(tool_query, OPENSEARCH_INDEX)
                    else:
                        tool_response = f"Unknown tool: {tool_function_name}"

                    with st.spinner("Running Sycamore query..."):
                        if isinstance(tool_response, str):
                            # We got a straight string response from the query plan, which means we can
                            # feed it back to the LLM directly.
                            tool_response_str = tool_response
                        elif isinstance(tool_response, sycamore.docset.DocSet):
                            # We got a DocSet.
                            # Note that this can be slow because the .take()
                            # actually runs the query.
                            tool_response_str = docset_to_string(tool_response)
                        else:
                            # Fall back to string representation.
                            tool_response_str = str(tool_response)
                        if not tool_response_str:
                            tool_response_str = "No results found for your query."
                except Exception as e:
                    st.error(f"Error running Sycamore query: {e}")
                    print(f"Error running Sycamore query: {e}")
                    # Print stack trace.
                    import traceback

                    traceback.print_exc()
                    tool_response_str = f"There was an error running your query: {e}"
                finally:
                    print(f"\nTool response: {tool_response_str}")
                    tool_response_message = ChatMessage(
                        {
                            "role": "tool",
                            "content": tool_response_str,
                            "tool_call_id": tool_call_id,
                            "name": tool_function_name,
                        }
                    )
                    st.session_state.messages.append(tool_response_message)
                    with st.expander("Tool response"):
                        st.write(f"```{tool_response_str}```")

            else:
                # No function call was made.
                assistant_message.show_content()
                if query_plan:
                    assistant_message.before_extras.append(ChatMessageExtra("Query plan", query_plan))
                if tool_response_str:
                    assistant_message.before_extras.append(
                        ChatMessageExtra("Tool response", f"```{tool_response_str}```")
                    )
                if query_id:
                    cmt = ChatMessageTraces("Query trace", query_id)
                    with st.expander("Query trace"):
                        cmt.show()
                    assistant_message.after_extras.append(cmt)
                break


def main():
    ray_init(address="auto")

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"

    if "use_cache" not in st.session_state:
        st.session_state.use_cache = True

    if "next_message_id" not in st.session_state:
        st.session_state.next_message_id = 0

    if "user_query" not in st.session_state:
        st.session_state.user_query = None

    if "messages" not in st.session_state:
        st.session_state.messages = [ChatMessage({"role": "assistant", "content": WELCOME_MESSAGE})]

    if "trace_dir" not in st.session_state:
        st.session_state.trace_dir = os.path.join(os.getcwd(), "traces")

    st.title(":airplane: Aryn NTSB Report Query Demo")
    st.toggle("Use RAG only", key="rag_only")
    show_messages()

    if prompt := st.chat_input("Ask me anything"):
        st.session_state.user_query = prompt

    while st.session_state.user_query is not None:
        do_query()


if __name__ == "__main__":
    main()
