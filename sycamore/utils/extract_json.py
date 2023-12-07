import re
import json


def extract_json(payload):
    pattern = r"```json([\s\S]*?)```"
    match = re.match(pattern, payload)
    if match:
        return json.loads(match.group(1))
    else:
        raise ValueError("JSON block not found in LLM response")
