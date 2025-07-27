# Start of config.py; will be inserted into other files to make a single bigquery UDF
import os
from google.cloud import storage, secretmanager


def get_secret(secret_id="aryn-api-key"):
    if (key := os.environ.get(id_to_env(secret_id), "")) != "":
        return key
    assert "DISABLE_SECRET_MANAGER" not in os.environ, f"DISABLE_SECRET_MANAGER is set, cannot get secret {secret_id}"
    client = secretmanager.SecretManagerServiceClient()
    project_id = storage.Client().project  # project isn't in secretmanager client :(
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def id_to_env(k):
    if k == "aryn-api-key":
        return "ARYN_BQ_API_KEY_1"
    if k == "aryn-api-key-2":
        return "ARYN_BQ_API_KEY_2"
    if k == "aryn-api-key-3":
        return "ARYN_BQ_API_KEY_3"
    if k == "aryn-overflow-prefix":
        return "ARYN_BQ_OVERFLOW_PREFIX"


configs = [
    {
        "key": get_secret(),
        "headers": {},
    },
    {
        "key": get_secret("aryn-api-key-2"),
        "headers": {},
    },
    {
        "key": get_secret(),
        "headers": {"X-Aryn-Asyncifier-Node": "1"},
    },
    {
        "key": get_secret("aryn-api-key-2"),
        "headers": {"X-Aryn-Asyncifier-Node": "1"},
    },
    {
        "key": get_secret("aryn-api-key-3"),
        "headers": {},
    },
]
# End of config.py; will be inserted into other files to make a single bigquery UDF
