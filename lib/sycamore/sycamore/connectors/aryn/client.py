import logging
from typing import Any

from aryn_sdk.client import Client

logger = logging.getLogger(__name__)


class ArynClient:
    def __init__(self, aryn_url: str, api_key: str):
        if aryn_url.endswith("/v1/storage"):
            aryn_url = aryn_url[: -len("/v1/storage")]
        self.client = Client(aryn_url=aryn_url, aryn_api_key=api_key)

    def list_docs(self, docset_id: str) -> list[str]:
        try:
            res = self.client.list_docs(docset_id=docset_id)
            docs = []
            for page in res.iter_page():
                for doc_md in page.value:
                    docs.append(doc_md.doc_id)
            return docs
        except Exception as e:
            raise ValueError(f"Error listing docs: {e}")

    def get_doc(self, docset_id: str, doc_id: str) -> dict[str, Any]:
        try:
            res = self.client.get_doc(docset_id=docset_id, doc_id=doc_id)
            doc = res.value.model_dump()
            logger.debug(f"Got doc {doc}")
            return doc
        except Exception as e:
            raise ValueError(f"Error getting doc {doc_id}: {e}")

    def create_docset(self, name: str) -> str:
        try:
            res = self.client.create_docset(name=name)
            return res.value.docset_id
        except Exception as e:
            raise ValueError(f"Error creating docset: {e}")
