#!/usr/bin/env python

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("querydemo/Query_Demo.py")
at.run()
assert not at.exception

at.chat_input[0].set_value("How many incidents were there in Washington?").run(timeout=120)
last_message = at.chat_message[-1]
print(type(last_message.children))
for child in last_message.children.values():
    print(f"child: {child}")
    if hasattr(child, "value"):
        print(f"   value: {child.value}")
