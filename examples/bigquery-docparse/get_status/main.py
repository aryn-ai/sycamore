"""
UDF for getting the status of an in-progress aysnc request
"""

# cloud functions requires this to be called main.py
# pip install functions_framework google-cloud-bigquery google-cloud-storage google-cloud-secret-manager

from contextlib import AbstractContextManager
import csv
import functions_framework
from google.cloud import storage
import httpx
import json
import os
import sys
from typing import Optional

from aryn_sdk.client.partition import (
    ARYN_DOCPARSE_URL,
    _process_config,
    _generate_headers,
    ArynConfig,
    _convert_sync_to_async_url,
    g_parameters,
)

sys.path.append(os.path.dirname(__file__) + "/..")
from config import configs, get_secret

OVERFLOW_PREFIX = get_secret("aryn-overflow-prefix")


def get_status(async_id):
    t = get_status_tup_except(async_id)
    return {"async_id": t[0], "new_async_id": t[1], "result": t[2], "err": t[3]}


def get_status_tup_except(async_id):
    try:
        return get_status_tup(async_id)
    except Exception as e:
        return (async_id, None, None, f"For {async_id} got exception: {e}")


def get_status_tup(async_id):  # -> old_async_id, new_async_id, result, error
    if async_id.startswith("aryn:"):
        config = configs[0]
        id_part = async_id
    elif len(parts := async_id.split("/")) == 2:
        config = configs[int(parts[0])]
        id_part = parts[1]
        assert id_part.startswith("aryn:")
    else:
        assert f"Unable to interpret {async_id}"

    key = config["key"]
    # print(f"Getting result with key={key}")
    with partition_file_async_result(id_part, aryn_api_key=key, extra_headers=config["headers"]) as response:
        headers = {}
        for name, value in response.headers.items():
            # print(f"  {name}: {value}")
            headers[name] = value

        if response.status_code == 202:  # still in progress
            print(f"{async_id} still in progress")
            return (async_id, async_id, None, None)
        if response.status_code == 404 and headers.get("x-aryn-asyncifier-msg") == "missing":
            return (async_id, None, None, f"Retrying {async_id} is not present")
        if response.status_code != 200 and headers.get("x-aryn-asyncifier-version") is not None:
            err = get_body(response, 8192)
            # permanent error; force a submit retry
            return (async_id, None, None, f"For {async_id} got permanent error: {err}")
        if response.status_code != 200:
            err = get_body(response, 8192)
            # transient error; don't retry submit
            return (async_id, async_id, None, f"For {async_id} got transient error: {err}")

        already_exists = False
        gcs_file = None
        uri = None
        large_len = 0
        data = b""
        for chunk in response.iter_bytes(1024 * 1024):
            if already_exists:
                continue

            if len(data) < 1024 * 1024:
                data += chunk
                continue

            if uri is None:
                uri = f"{OVERFLOW_PREFIX}{async_id}"
                print(f"Got large result, uploading to {uri}")
                bucket, object = uri.removeprefix("gs://").split("/", 1)
                blob = storage.Client().bucket(bucket).blob(object)
                if blob.exists():
                    print("and it already exists")
                    already_exists = True
                    continue

                gcs_file = blob.open("wb")
                large_len = len(data)
                gcs_file.write(data)

            assert gcs_file is not None
            gcs_file.write(chunk)
            large_len += len(chunk)

        if already_exists:
            assert gcs_file is None and uri is not None
            return (async_id, async_id, uri.encode(), None)

        if gcs_file is not None:
            assert uri is not None
            gcs_file.close()
            print(f"Wrote {large_len} bytes to gcs")
            return (async_id, async_id, uri.encode(), None)

        assert uri is None and large_len == 0
        return (async_id, async_id, data, None)


def get_body(response, max_bytes):
    ret = b""
    for chunk in response.iter_bytes(chunk_size=max_bytes):
        if len(ret) < max_bytes:
            ret = ret + chunk
    return ret


def upload_large(data, uri):
    bucket, object = uri.removeprefix("gs://").split("/", 1)
    blob = storage.Client().bucket(bucket).blob(object)
    if blob.exists():
        print(f"Noop; blob {uri} already exists")
        return
    blob.upload_from_string(data, content_type="application/json", if_generation_match=0)
    print(f"Uploaded blob {uri}")


# from aryn_sdk.client.partition import partition_file_async_result
# local copy to rewrite to fully incremental


def partition_file_async_result(
    task_id: str,
    *,
    aryn_api_key: Optional[str] = None,
    aryn_config: Optional[ArynConfig] = None,
    ssl_verify: bool = True,
    async_result_url: Optional[str] = None,
    extra_headers={},
) -> AbstractContextManager[httpx.Response]:
    """
    Get the results of an asynchronous partitioning task by task_id. Meant to be used with
    `partition_file_async_submit`.

    For examples of usage see README.md

    Raises a `PartitionTaskNotFoundError` if the not task with the task_id can be found.

    Returns:
        A dict containing "status" and "status_code". When "status" is "done", the returned dict also contains "result"
        which contains what would have been returned had `partition_file` been called directly. "status" can be "done"
        or "pending".

        Unlike `partition_file`, this function does not raise an Exception if the partitioning failed.
    """
    if not async_result_url:
        async_result_url = _convert_sync_to_async_url(ARYN_DOCPARSE_URL, "/result", truncate=True)

    aryn_config = _process_config(aryn_api_key, aryn_config)

    assert async_result_url is not None
    specific_task_url = f"{async_result_url.rstrip('/')}/{task_id}"
    headers = _generate_headers(aryn_config.api_key())
    headers.update(extra_headers)
    return httpx.stream("GET", specific_task_url, params=g_parameters, headers=headers, verify=ssl_verify)


@functions_framework.http
def get_status_entrypoint(request):
    """
    Cloud Function HTTP entry point for the BigQuery remote UDF.
    BigQuery sends a POST request with a JSON payload in the format:
    {"calls": [["async_id_1"], ["async_id_2"], ...]}
    The function should return a JSON payload in the format:
    {"replies": [result_for_async_id_1, result_for_async_id_2, ...]}
    """
    try:
        # Get the JSON request body
        request_json = request.get_json(silent=True)
        print(f"Received raw request: {request_json}")

        if not request_json or "calls" not in request_json:
            return (
                json.dumps({"error": "Invalid request format. Expected {'calls': [...]}"}),
                400,
                {"Content-Type": "application/json"},
            )

        replies = []
        # BigQuery sends an array of arrays, where each inner array contains the parameters
        # For a single parameter UDF, it will be `[["value1"], ["value2"]]`
        for call_args in request_json["calls"]:
            if not isinstance(call_args, list) or not call_args:
                replies.append({"error": "Invalid call argument format"})
                continue

            async_id = call_args[0]  # Assuming async_id is the first (and only) parameter

            try:
                status_data = get_status(async_id)
                replies.append(status_data)
            except Exception as e:
                print(f"Error processing async_id {async_id}: {e}")
                replies.append({"async_id": async_id, "new_async_id": None, "result": "ERROR", "err": str(e)})

        return json.dumps({"replies": replies}), 200, {"Content-Type": "application/json"}

    except Exception as e:
        print(f"Global error processing request: {e}")
        return json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"}


if __name__ == "__main__":
    if True:
        print(get_status_tup("4/aryn:t-17i7taigfxcvtq96mk08ucx"))
        # print(get_status_tup("0/aryn:t-9aoh7233tlcemjq1mmnvmvy"))
        # print(get_status_tup("1/aryn:t-9aoh7233tlcemjq1mmnvmvy"))
        # print(get_status_tup("1/aryn:t-vpydotrg8zxh7rmvw9cmheo"))
    if False:
        a = get_status_tup("aryn:t-h2v96nffcql87q5ppuz3ix1")
        b = get_status_tup("0/aryn:t-h2v96nffcql87q5ppuz3ix1")
        assert a[2] == b[2] and len(a[2]) > 1000
        print("Passed nothing = 0/")
        c = get_status_tup("1/aryn:t-h2v96nffcql87q5ppuz3ix1")
        print(c)
        exit(0)
    if False:
        with open("/home/eric/Downloads/script_job_942a68acffc61a56e0aa71c7f2412d8c_0.csv") as f:
            reader = csv.reader(f)
            for r in reader:
                if r[1] == "aryn:t-q5xs3wcy9bknxy5bez1sqrz":
                    print("Big thing, skip")
                    continue
                print(r[1])
                orig_async_id, new_async_id, result, err = get_status_tup(r[1])
                if result is None:
                    result = ""
                print(f"RES {orig_async_id} {new_async_id} {len(result)} {str(err)[0:40]}")
