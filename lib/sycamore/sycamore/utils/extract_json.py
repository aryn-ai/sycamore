import re
import json
from json import JSONDecodeError
from typing import Any


def extract_json(payload: str) -> Any:
    """Given the provided payload, extract the JSON block from it."""

    # Replace Python's None with JSON's null, being careful to not replace
    # strings that might contain "None" as part of their content
    payload = re.sub(r":\s*None\b", ": null", payload)

    try:
        return json.loads(payload)
    except (ValueError, TypeError, JSONDecodeError) as exc:
        # Sometimes the LLM makes up an escape code. In that case,
        # replace the escape char with its representation (e.g. \\x07)
        # and recurse.
        if isinstance(exc, JSONDecodeError) and "Invalid \\escape" in exc.msg:
            c = payload[exc.pos]
            payload = payload[: exc.pos] + repr(c)[1:-1] + payload[exc.pos + 1 :]
            return extract_json(payload)
        # It is possible that the LLM response includes a code block with JSON data.
        # Pull the JSON content out from it.
        pattern = r"```json([\s\S]*?)```"
        match = re.match(pattern, payload)
        if match:
            return json.loads(match.group(1))
        else:
            raise ValueError("JSON block not found in LLM response: " + str(payload)) from exc
