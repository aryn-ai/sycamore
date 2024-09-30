import os
import tempfile
from typing import Annotated, List, Optional

from fastapi import FastAPI, Path
from pydantic import BaseModel
from sycamore.query import SycamoreQueryClient

import queryserver.util as util


app = FastAPI()


CACHE_PATH = os.getenv("QUERYSERVER_CACHE_PATH", os.path.join(tempfile.gettempdir(), "queryserver_cache"))
LLM_CACHE_PATH = os.getenv("QUERYSERVER_LLM_CACHE_PATH", os.path.join(tempfile.gettempdir(), "queryserver_llm_cache"))

client = SycamoreQueryClient(s3_cache_path=LLM_CACHE_PATH, cache_dir=CACHE_PATH)


def get_sycamore_query_client(
    s3_cache_path: Optional[str] = None, trace_dir: Optional[str] = None
) -> SycamoreQueryClient:
    pass


class Index(BaseModel):
    index_id: str
    description: Optional[str] = None
    columns: List[str]


class Query(BaseModel):
    query: str
    index: str
    examples: Optional[List[str]] = None




@app.get("/v1/indices")
async def list_indices() -> List[Index]:
    indices = get_opensearch_indices()

    return {x for x in SycamoreQueryClient().get_opensearch_incides() if not x.startswith(".")}

    return [
        Index(index_id="products", columns=["name", "description", "price", "tax"]),
        Index(index_id="customers", columns=["name", "email", "phone"]),
    ]


@app.get("/v1/index/{index_id}")
async def get_index(
    index_id: Annotated[str, Path(title="The ID of the index to get")],
):
    return Index(index_id=index_id, columns=["name", "description", "price", "tax"])


@app.post("/v1/plan")
async def generate_plan(
    index_id: Annotated[str, Path(title="The ID of the index to get")],
):
    return Index(index_id=index_id, columns=["name", "description", "price", "tax"])
