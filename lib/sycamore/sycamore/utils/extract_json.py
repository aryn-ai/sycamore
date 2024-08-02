import re
import json
from json import JSONDecodeError


def extract_json(payload):
    try:
        return json.loads(payload)
    except (ValueError, TypeError, JSONDecodeError):
        pattern = r"```json([\s\S]*?)```"
        match = re.match(pattern, payload)
        if match:
            return json.loads(match.group(1))
        else:
            raise ValueError("JSON block not found in LLM response: " + str(payload))
