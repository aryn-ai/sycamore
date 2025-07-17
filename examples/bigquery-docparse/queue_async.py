"""
UDF for queueing an async request to Aryn DocParse
"""
import datetime
import os
import random
from google.cloud import storage, secretmanager
from aryn_sdk.client.partition import partition_file_async_submit, partition_file_async_result, ArynConfig

sys.path.append(os.path.dirname(__file__) + "/.."); from config import *

def queue_async(uri, async_id, config_list):
    possible_config_nums = [int(k) for k in config_list.split(",")]
    config_num = possible_config_nums[random.randint(0, len(possible_config_nums) - 1)]
    config = configs[config_num]

    if async_id is None:
        bytes = get_pdf(uri)
        key, headers = config["key"], config["headers"]
        print(f"Submitting using key {key}; headers {headers}")
        submit = partition_file_async_submit(
            bytes, aryn_api_key=config["key"], extract_table_structure=True, filename=uri, extra_headers=headers
        )
        return f"{config_num}/{submit['task_id']}"
    else:
        return async_id


def get_pdf(uri):
    bucket, object = uri.removeprefix("gs://").split("/", 1)
    return storage.Client().bucket(bucket).blob(object).download_as_bytes()

if __name__ == "__main__":
    sa = queue_async("gs://eric-aryn-test-bucket/visit_aryn.pdf", None, "2")
    print(f"submitted as {sa}")
