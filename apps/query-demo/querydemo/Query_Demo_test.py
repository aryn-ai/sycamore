#!/usr/bin/env python

import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ChatMessage

at = AppTest.from_file("querydemo/Query_Demo.py")
at.run()
assert not at.exception


def get_last_message() -> ChatMessage:
    return at.chat_message[-1]


def get_message_markdown(message) -> str:
    retval = None
    for child in message.children.values():
        if child.type == "markdown":
            if retval is None:
                retval = child.value
            else:
                retval += child.value
    if retval is None:
        raise ValueError("No markdown found in message")
    return retval


def do_query(query: str, timeout: int = 120) -> ChatMessage:
    at.chat_input[0].set_value(query).run(timeout=timeout)
    return get_last_message()


def run_tests():
    result = do_query("How many incidents were there in Washington?")
    md = get_message_markdown(result)
    assert "3" in md or "three" in md

    result = do_query("What types of aircrafts were involved in accidents in California?")
    md = get_message_markdown(result)
    assert "<TableCell>Aircraft Type</TableCell>" in md
    assert "<TableCell>Accidents</TableCell>" in md
    assert "<TableCell>Cessna 172</TableCell>" in md
    assert "<TableCell>Cessna 180K</TableCell>" in md
    assert "<TableCell>Cessna 195A</TableCell>" in md
    assert "<TableCell>Cessna 414</TableCell>" in md
    assert "<TableCell>Cessna T21ON</TableCell>" in md
    assert "<TableCell>Piper PA-28-180</TableCell>" in md
    assert "<TableCell>Piper PAZ8R</TableCell>" in md
    print("All tests passed.")

run_tests()
