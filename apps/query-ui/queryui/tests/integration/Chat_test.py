#!/usr/bin/env python

# This is a set of tests for Chat.py.
# Run with:
#   poetry run queryui/main.py --test [pytest-args]
# where [pytest-args] are any additional arguments you want to pass to pytest.
#
# Recommended usage:
#   poetry run python queryui/main.py --test --show-capture=stdout

import json
import tempfile
from unittest.mock import patch
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block, Expander, Json, Node, ChatMessage, ElementTree


# This is the index that must exist locally that is used for testing.
# It is assumed that this index contains the NTSB dataset populated by
# `loaddata.py` in the `queryui` directory.
TEST_INDEX = "const_ntsb"


@pytest.fixture
def app() -> Generator[AppTest, None, None]:
    """Fixture to start the app."""
    with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as llm_cache_dir:
        print(f"Test cache dir: {cache_dir}")
        print(f"Test LLM cache dir: {llm_cache_dir}")
        with patch(
            "sys.argv", ["Chat.py", "--index", TEST_INDEX, "--cache-dir", cache_dir, "--llm-cache-dir", llm_cache_dir]
        ):
            at = AppTest.from_file("../../pages/Chat.py").run()
            yield at


def get_last_message(at: AppTest) -> ChatMessage:
    """Return the last message in the chat."""
    return at.chat_message[-1]


def do_query(at: AppTest, query: str, timeout: int = 120) -> ChatMessage:
    """Run a query and return the last message result."""
    at.chat_input[0].set_value(query).run(timeout=timeout)
    return get_last_message(at)


def walk(node: Node, fn: Callable[[Node, Any], Any], initial: Optional[Any] = None) -> Any:
    """Walk the node tree, running the given function on each node with the current value of the accumulator."""
    retval = fn(node, initial)
    if hasattr(node, "children"):
        for child in node.children.values():
            retval = walk(child, fn, retval)
    return retval


def expect_any(node: Node, fn: Callable[[Node], bool]):
    """Expect that at least one node in the tree satisfies the given function."""

    def collect(node, retval):
        retval.append(fn(node))
        return retval

    assert any(walk(node, collect, []))


def expect_all(node: Node, fn: Callable[[Node], bool]):
    """Expect that all nodes in the tree satisfy the given function."""

    def collect(node, retval):
        retval.append(fn(node))
        return retval

    assert all(walk(node, collect, []))


def expect_none(node: Node, fn: Callable[[Node], bool]):
    """Expect that no nodes in the tree satisfy the given function."""

    def collect(node, retval):
        retval.append(fn(node) is False)
        return retval

    assert all(walk(node, collect, []))


def get_markdown(node: Node) -> str:
    """Return the markdown content of the given node tree."""

    def markdown(node: Node, retval: str) -> str:
        if node.type == "markdown":
            assert hasattr(node, "value")
            retval += node.value
        elif node.type == "arrow_data_frame":
            assert hasattr(node, "value")
            retval += node.value.to_markdown()
        elif node.type == "header":
            assert hasattr(node, "value")
            retval += node.value
        elif node.type == "subheader":
            assert hasattr(node, "value")
            retval += node.value
        return retval + "\n\n"

    return walk(node, markdown, "")


@pytest.fixture
def block_tree() -> Block:
    """Fixture to create a test block tree."""

    class TestBlock(Block):
        def __init__(self, label: str, root: ElementTree, children: Dict[int, Node] = {}):
            super().__init__(proto=None, root=root)
            self.label = label
            self.children = children

        def __str__(self):
            return f"<TestBlock {self.label}>"

    root = ElementTree()
    return TestBlock(
        "top",
        root,
        {
            0: TestBlock("a", root),
            1: TestBlock(
                "b",
                root,
                {
                    3: TestBlock("b1", root),
                    4: TestBlock("b2", root),
                },
            ),
            2: TestBlock("c", root),
        },
    )


def test_walk(block_tree):
    """Test the walk function."""
    result = []

    def print_node(node, count):
        nonlocal result
        result.append(str(node) + ":" + str(count))
        return count + 1

    assert walk(block_tree, print_node, 0) == 6
    assert result == [
        "<TestBlock top>:0",
        "<TestBlock a>:1",
        "<TestBlock b>:2",
        "<TestBlock b1>:3",
        "<TestBlock b2>:4",
        "<TestBlock c>:5",
    ]


def test_expect_any(block_tree):
    """Test the expect_any function."""
    top = block_tree
    expect_any(top, lambda n: n.label == "a")
    expect_any(top, lambda n: n.label == "b")
    expect_any(top, lambda n: n.label == "c")
    expect_any(top, lambda n: n.label == "a" or n.label == "b2")
    with pytest.raises(AssertionError):
        expect_any(top, lambda n: n.label == "d")


def test_expect_all(block_tree):
    """Test the expect_all function."""
    top = block_tree
    expect_all(top, lambda n: n.label)
    with pytest.raises(AssertionError):
        expect_all(top, lambda n: n.label == "a")


def test_expect_none(block_tree):
    """Test the expect_none function."""
    top = block_tree
    expect_none(top, lambda n: n.label == "d")
    with pytest.raises(AssertionError):
        expect_none(top, lambda n: n.label == "a")


def test_app_starts(app):
    """Test that the app starts without raising an exception."""
    assert app.title[0].value == "Sycamore Query Chat"
    assert not app.exception


def check_query_result(result: ChatMessage) -> Tuple[str, str]:
    """Check that the query result satisfies the expected format.

    Returns the markdown content of the result text and the query traces.
    """

    # Should be no exception nodes.
    expect_none(result, lambda x: x.type == "exception")

    # Query plan should be present.
    expanders: List[Expander] = [
        x for x in result.children.values() if x.type == "expander" and isinstance(x, Expander)
    ]
    # assert all([isinstance(x, Expander) for x in expanders])
    assert expanders[0].label == "Query plan"
    assert expanders[0].children[0].type == "json"
    assert isinstance(expanders[0].children[0], Json)
    query_plan_json = json.loads(expanders[0].children[0].value)
    assert isinstance(query_plan_json, dict)
    assert "query" in query_plan_json
    assert set(query_plan_json.keys()) == {"query", "nodes", "result_node", "llm_prompt", "llm_plan"}

    # Sycamore query result should be present.
    assert expanders[1].label == "Sycamore query result"
    assert expanders[1].children[0].type == "markdown"

    # Query traces should be present.
    assert expanders[2].label == "Query trace"
    print(expanders[2])

    assert expanders[2].children[0].type == "vertical"
    assert expanders[2].children[0].children[0].children[0].label == "Node data"  # type: ignore
    assert expanders[2].children[0].children[0].children[1].label == "Query plan"  # type: ignore
    assert expanders[2].children[0].children[0].children[0].children[1].type == "markdown"
    return get_markdown(result), get_markdown(expanders[2].children[0].children[0].children[0])


def test_count_query(app):
    """Test a basic count query."""
    result = do_query(app, "How many incidents were there in Washington?")
    md, traces = check_query_result(result)

    md = get_markdown(result)
    assert "3" in md or "three" in md

    assert "Node 0 - QueryDatabase" in traces
    assert "Yelm" in traces
    assert "Dallesport" in traces
    assert "Kent" in traces
    assert "WPR23LA086" in traces
    assert "WPR23LA089" in traces
    assert "WPR2BLA101" in traces

    assert "Node 1 - Count" in traces
    assert "Node type Count does not produce document traces" in traces


def test_aircraft_types_query(app):
    """Test a query for aircraft types."""
    result = do_query(app, "What types of aircrafts were involved in accidents in California?")
    md, _ = check_query_result(result)

    cessnas = ["Cessna 172", "Cessna 180K", "Cessna 195A", "Cessna 414", "Cessna T21ON"]
    piper = ["Piper PA-28-180", "Piper PAZ8R"]

    assert sum(1 for c in cessnas if c in md) >= 3
    assert any(p in md for p in piper)


def test_rag_count_query(app):
    """Test a RAG count query."""
    app.toggle[0].set_value(True).run()
    result = do_query(app, "How many incidents were there in Washington?")
    expanders = [x for x in result.children.values() if x.type == "expander"]
    assert expanders[0].label == "RAG query"
    assert expanders[1].label == "Sycamore query result"

    md = get_markdown(result)
    assert "incidents" in md
