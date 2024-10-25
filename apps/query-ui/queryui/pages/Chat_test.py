#!/usr/bin/env python

# This is a set of tests for Chat.py.
# Run with:
#   poetry run queryui/main.py --test [pytest-args]
# where [pytest-args] are any additional arguments you want to pass to pytest.

import json
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ChatMessage


TEST_INDEX = "const_ntsb"


@pytest.fixture
def app() -> AppTest:
    """Fixture to start the app."""
    with patch("sys.argv", ["Chat.py", "--index", TEST_INDEX]):
        at = AppTest.from_file("Chat.py").run()
        yield at


def get_last_message(at: AppTest) -> ChatMessage:
    """Return the last message in the chat."""
    return at.chat_message[-1]


def get_message_markdown(message: ChatMessage) -> str:
    """Return the markdown content of the given chat message."""
    retval = ""
    for child in message.children.values():
        if child.type == "markdown":
            assert hasattr(child, "value")
            retval += child.value
        elif child.type == "arrow_data_frame":
            assert hasattr(child, "value")
            retval += child.value.to_markdown()
    if not retval:
        raise ValueError("No markdown found in message")
    return retval


def do_query(at: AppTest, query: str, timeout: int = 120) -> ChatMessage:
    """Run a query and return the last message result."""
    at.chat_input[0].set_value(query).run(timeout=timeout)
    return get_last_message(at)


def test_app_starts(app):
    assert app.title[0].value == "Sycamore Query Chat"
    assert not app.exception


def test_count_query(app):
    result = do_query(app, "How many incidents were there in Washington?")
    md = get_message_markdown(result)
    assert "3" in md or "three" in md

    # Check for query plan.
    expanders = [x for x in result.children.values() if x.type == "expander"]
    assert expanders[0].label == "Query plan"
    assert expanders[0].children[0].type == "json"
    query_plan_json = json.loads(expanders[0].children[0].value)
    assert isinstance(query_plan_json, dict)
    assert "query" in query_plan_json
    assert set(query_plan_json.keys()) == {"query", "nodes", "result_node", "llm_prompt", "llm_plan"}

    # Check for Sycamore query result.
    assert expanders[1].label == "Sycamore query result"
    assert expanders[1].children[0].type == "markdown"
    assert "3" in expanders[1].children[0].value or "three" in expanders[1].children[0].value

    # Check for query traces.
    assert expanders[2].label == "Query trace"
    def walk(node, depth=0):
        print(f"{'  ' * depth}Node: {node}")
        if hasattr(node, "children"):
            for child in node.children:
                walk(child, depth + 1)

    walk(expanders[2])
    assert expanders[2].children[0].type == "bbvertical"


def test_aircraft_types_query(app):
    result = do_query(app, "What types of aircrafts were involved in accidents in California?")

    md = get_message_markdown(result)
    assert "Cessna 172" in md
    assert "Cessna 180K" in md
    assert "Cessna 195A" in md
    assert "Cessna 414" in md
    assert "Cessna T21ON" in md
    assert "Piper PA-28-180" in md
    assert "Piper PAZ8R" in md


def test_rag_count_query(app):
    app.toggle[0].set_value(True).run()
    result = do_query(app, "How many incidents were there in Washington?")
    expanders = [x for x in result.children.values() if x.type == "expander"]
    assert expanders[0].label == "RAG query"
    assert expanders[1].label == "Sycamore query result"

    md = get_message_markdown(result)
    assert "incidents" in md
