import re
import json
from json import JSONDecodeError
from typing import Any


def extract_json(payload: str) -> Any:
    """Given the provided payload, extract the JSON block from it."""

    try:
        return json.loads(payload)
    except (ValueError, TypeError, JSONDecodeError) as exc:
        # It is possible that the LLM response includes a code block with JSON data.
        # Pull the JSON content out from it.
        pattern = r"```json([\s\S]*?)```"
        match = re.match(pattern, payload)
        if match:
            return json.loads(match.group(1))
        else:
            raise ValueError("JSON block not found in LLM response: " + str(payload)) from exc
