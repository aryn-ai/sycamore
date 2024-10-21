# This is a FastAPI server that wraps Sycamore Query in a REST service.
#
# Run with:
#   poetry run fastapi dev queryserver/main.py

import os
import tempfile
from typing import Annotated, List, Optional

from fastapi import FastAPI, Path
from pydantic import BaseModel
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.result import SycamoreQueryResult
from sycamore.query.schema import OpenSearchSchema

import queryserver.util as util


app = FastAPI()


CACHE_PATH = os.getenv("QUERYSERVER_CACHE_PATH", os.path.join(tempfile.gettempdir(), "queryserver_cache"))
LLM_CACHE_PATH = os.getenv("QUERYSERVER_LLM_CACHE_PATH", os.path.join(tempfile.gettempdir(), "queryserver_llm_cache"))

sqclient = SycamoreQueryClient(llm_cache_dir=LLM_CACHE_PATH, cache_dir=CACHE_PATH)


class Index(BaseModel):
    """Represents an index that can be queried."""

    index: str
    description: Optional[str] = None
    index_schema: OpenSearchSchema


class Query(BaseModel):
    """Query an index with a given natural language query string."""

    query: str
    index: str


def get_index_schema(index: str) -> OpenSearchSchema:
    """Get the schema for the given index."""
    return sqclient.get_opensearch_schema(index)


@app.get("/v1/indices")
async def list_indices() -> List[Index]:
    """List all available indices."""

    retval = []
    indices = util.get_opensearch_indices()
    for index in indices:
        index_schema = sqclient.get_opensearch_schema(index)
        retval.append(Index(index=index, index_schema=index_schema))
    return retval


@app.get("/v1/index/{index}")
async def get_index(
    index: Annotated[str, Path(title="The ID of the index to get")],
) -> Index:
    """Return details on the given index."""

    schema = get_index_schema(index)
    return Index(index=index, index_schema=schema)


@app.post("/v1/plan")
async def generate_plan(query: Query) -> LogicalPlan:
    """Generate a query plan for the given query, but do not run it."""

    plan = sqclient.generate_plan(query.query, query.index, util.get_schema(sqclient, query.index))
    return plan


@app.post("/v1/plan/run")
async def run_plan(plan: LogicalPlan) -> SycamoreQueryResult:
    """Run the provided query plan."""

    return sqclient.run_plan(plan)


@app.post("/v1/query")
async def run_query(query: Query) -> SycamoreQueryResult:
    """Generate a plan for the given query, run it, and return the result."""

    plan = sqclient.generate_plan(query.query, query.index, util.get_schema(sqclient, query.index))
    return sqclient.run_plan(plan)
