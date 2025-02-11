import logging
from typing import Any

import requests

from sycamore.decorators import experimental

logger = logging.getLogger(__name__)


@experimental
class ArynClient:
    def __init__(self, aryn_url: str, api_key: str):
        self.aryn_url = aryn_url
        self.api_key = api_key

    def list_docs(self, docset_id: str) -> list[str]:
        try:
            response = requests.get(
                f"{self.aryn_url}/docsets/{docset_id}/docs", headers={"Authorization": f"Bearer {self.api_key}"}
            )
            items = response.json()["items"]
            return [item["doc_id"] for item in items]
        except Exception as e:
            raise ValueError(f"Error listing docs: {e}")

    def get_doc(self, docset_id: str, doc_id: str) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.aryn_url}/docsets/{docset_id}/docs/{doc_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Error getting doc {doc_id}, received {response.status_code} {response.text} {response.reason}"
                )
            doc = response.json()
            if doc is None:
                raise ValueError(f"Received None for doc {doc_id}")
            logger.debug(f"Got doc {doc}")
            return doc
        except Exception as e:
            raise ValueError(f"Error getting doc {doc_id}: {e}")

    def create_docset(self, name: str) -> str:
        try:
            response = requests.post(
                f"{self.aryn_url}/docsets", json={"name": name}, headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.json()["docset_id"]
        except Exception as e:
            raise ValueError(f"Error creating docset: {e}")
