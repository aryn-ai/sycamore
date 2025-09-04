import re
import json
from json import JSONDecodeError
from typing import Any


def extract_json(payload: str, verbose=False) -> Any:
    """Given the provided payload, extract the JSON block from it."""

    orig_payload = payload
    if verbose:
        print(f"--------------------Extract Json--------------------\n{payload}\n")
    # It is possible that the LLM response includes a code block with JSON data.
    # Pull the JSON content out from it.
    pattern = r"(^|.*\n)```json\n([\s\S]*?\n)```"
    match = re.match(pattern, payload, re.DOTALL)
    if match:
        payload = match.group(2)
        if verbose:
            print(f"  ----------------------remove ```json ``` wrapper -----------\n{payload}\n")
    elif verbose and "```json" in payload:
        print(f"  --------------------did not remove ```json from payload")
            
    # Replace Python's None with JSON's null, being careful to not replace
    # strings that might contain "None" as part of their content
    p2 = re.sub(r":\s*None\b", ": null", payload)
    if p2 != payload:
        payload = p2
        if verbose:
            print(f"  ----------------------replace Python None -----------------\n{payload}\n")
            
    try:
        return json.loads(payload)
    except JSONDecodeError as exc:
        # Sometimes the LLM makes up an escape code. In that case,
        # replace the escape char with its representation (e.g. \\x07)
        # and recurse.
        if "Invalid \\escape" in exc.msg:
            c = payload[exc.pos]
            payload = payload[: exc.pos] + repr(c)[1:-1] + payload[exc.pos + 1 :]
            if verbose:
                print(f"  --------------------------replace invalid escape {c} --------\n{payload}")
            return extract_json(payload, verbose=verbose)
        else:
            raise ValueError("JSON block not found in LLM response: " + str(orig_payload)) from exc
