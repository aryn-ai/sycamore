# This file contains utility functions enabling agentic chat with the Sycamore Query UI.

import datetime
from html.parser import HTMLParser
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


import queryui.util as util

from sycamore.query.result import SycamoreQueryResult


# The agent LLM needs a prompt that understands how to render MDX output with the appropriate
# set of JSX components. This prompt should be appended to the system prompt to ensure
# that the agent knows how to render the output.
MDX_SYSTEM_PROMPT = """All your responses should be in MDX, which is Markdown augmented with JSX
components.  As an example, your response could include a table of data, a link to a document, or a
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


class ChatMessageExtra:
    """Extra content to show before or after a ChatMessage."""

    def __init__(self, name: str, content: Optional[Any] = None):
        self.name = name
        self.content = content

    def show(self):
        """Show this ChatMessageExtra."""
        st.write(self.content)


class ChatMessageTraces(ChatMessageExtra):
    """A ChatMessageExtra subclass for showing query traces."""

    def __init__(self, name: str, result: SycamoreQueryResult):
        super().__init__(name)
        self.result = result

    def show(self):
        util.show_query_traces(self.result)


class ChatMessage:
    """Represents a single message in a chat session with the agent.

    Args:
        message: The content of the message.
        before_extras: A list of ChatMessageExtra objects to display before the message content.
        after_extras: A list of ChatMessageExtra objects to display after the message content.
    """

    def __init__(
        self,
        message: Optional[Dict[str, Any]] = None,
        before_extras: Optional[List[ChatMessageExtra]] = None,
        after_extras: Optional[List[ChatMessageExtra]] = None,
    ):
        assert "next_message_id" in st.session_state and isinstance(st.session_state.next_message_id, int)
        self.message_id = st.session_state.next_message_id
        st.session_state.next_message_id += 1
        self.timestamp = datetime.datetime.now()
        self.message = message or {}
        self.before_extras = before_extras or []
        self.after_extras = after_extras or []
        self.widget_key = 0

    def next_key(self) -> str:
        """Streamlit requires widgets to have unique keys. Used to generate a new key for each widget."""
        key = f"{self.message_id}-{self.widget_key}"
        self.widget_key += 1
        return key

    def show(self):
        """Show this ChatMessage."""
        self.widget_key = 0
        with st.chat_message(self.message.get("role", "assistant")):
            self.show_content()

    def show_content(self):
        """Show just the contents of this ChatMessage, without the surrounding role container."""
        for extra in self.before_extras:
            with st.expander(extra.name):
                extra.show()
        if self.message.get("content"):
            content = self.message.get("content")
            self.render_mdx(content)
        for extra in self.after_extras:
            with st.expander(extra.name):
                extra.show()

    def button(self, label: str, **kwargs):
        """Return a button widget."""
        return st.button(label, key=self.next_key(), **kwargs)

    def render_mdx(self, text: str):
        """Render the given text as MDX."""
        print(f"Rendering MDX:\n{text}\n")
        renderer = MDXRenderer(self)
        renderer.feed(text)

    def render_markdown(self, text: str):
        """Render markdown text."""
        st.markdown(text)

    def render_table(self, columns: List[str], rows: List[List[str]]):
        """Render a table with the given columns and rows."""
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
        """Render a preview of the PDF document at the given path."""
        with st.container(border=True):
            col1, col2 = st.columns([0.75, 0.25])
            with col1:
                st.text("📄 " + path)
            with col2:
                if st.button("View document", key=self.next_key()):
                    util.show_pdf_preview(path)

    def render_suggested_query(self, query: str):
        """Render a suggested query."""
        if self.button(query):
            st.session_state.user_query = query

    def render_map(self, data: List[Tuple[float, float]]):
        """Render a map with the given (latitude, longitude) points labeled."""
        df = pd.DataFrame(data, columns=["lat", "lon"])
        st.map(df, color="#00ff00", size=100)

    def to_dict(self) -> Optional[Dict[str, Any]]:
        """Return a dict that can be sent to an LLM representing the content of this message."""
        return self.message

    def __str__(self):
        return f"{self.message.get('role', 'assistant')}: {self.message.get('content')}"


class MDXRenderer(HTMLParser):
    """Render MDX, which is Markdown with embedded JSX components.

    We instruct the LLM to generate MDX output, allowing us to embed tables, maps, and other
    components in the rendered output. This class is used to parse the MDX and invoke the appropriate
    ChatMessage methods to render the embedded components.
    """

    def __init__(self, chat_message: ChatMessage):
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
