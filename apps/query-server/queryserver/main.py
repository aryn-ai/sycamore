# This is a FastAPI server that wraps Sycamore Query in a REST service.
#
# Run with:
#   poetry run fastapi dev queryserver/main.py

import os
import tempfile
from typing import Annotated, Any, Dict, List, Optional

from fastapi import FastAPI, Path
from pydantic import BaseModel
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan

import queryserver.util as util


app = FastAPI()


CACHE_PATH = os.getenv("QUERYSERVER_CACHE_PATH", os.path.join(tempfile.gettempdir(), "queryserver_cache"))
LLM_CACHE_PATH = os.getenv("QUERYSERVER_LLM_CACHE_PATH", os.path.join(tempfile.gettempdir(), "queryserver_llm_cache"))

sqclient = SycamoreQueryClient(s3_cache_path=LLM_CACHE_PATH, cache_dir=CACHE_PATH)


def get_sycamore_query_client(
    s3_cache_path: Optional[str] = None, trace_dir: Optional[str] = None
) -> SycamoreQueryClient:
    pass


class IndexSchemaField(BaseModel):
    """Represents a single field in an index schema."""

    field_type: str
    examples: Optional[List[str]]


class IndexSchema(BaseModel):
    """Represents the schema for a given index."""

    fields: Dict[str, IndexSchemaField] = {}


class Index(BaseModel):
    """Represents an index that can be queried."""

    index: str
    description: Optional[str] = None
    schema: IndexSchema


class Query(BaseModel):
    """Query an index with a given natural language query string."""

    query: str
    index: str


class QueryResult(BaseModel):
    """Represents the result of a query operation."""

    plan: LogicalPlan
    result: Any


def get_index_schema(index: str) -> IndexSchema:
    """Get the schema for the given index."""

    index_schema = IndexSchema()
    schema = sqclient.get_opensearch_schema(index)
    for field in schema:
        index_schema.fields[field] = IndexSchemaField(field_type=schema[field][0], examples=schema[field][1])
    return index_schema


@app.get("/v1/indices")
async def list_indices() -> List[Index]:
    """List all available indices."""

    retval = []
    indices = util.get_opensearch_indices()
    for index in indices:
        index_schema = IndexSchema()
        schema = sqclient.get_opensearch_schema(index)
        for field in schema:
            index_schema.fields[field] = IndexSchemaField(field_type=schema[field][0], examples=schema[field][1])
        retval.append(Index(index=index, schema=index_schema))
    return retval


@app.get("/v1/index/{index}")
async def get_index(
    index: Annotated[str, Path(title="The ID of the index to get")],
) -> Index:
    """Return details on the given index."""

    schema = get_index_schema(index)
    return Index(index=index, schema=schema)


@app.post("/v1/plan")
async def generate_plan(query: Query) -> LogicalPlan:
    """Generate a query plan for the given query, but do not run it."""

    plan = sqclient.generate_plan(
        query.query, query.index, util.get_schema(sqclient, query.index), examples=query.examples
    )
    return plan


@app.post("/v1/plan/run")
async def run_plan(plan: LogicalPlan) -> QueryResult:
    """Run the provided query plan."""

    _, result = sqclient.run_plan(plan)
    return QueryResult(plan=plan, result=result)


@app.post("/v1/query")
async def run_query(query: Query) -> QueryResult:
    """Generate a plan for the given query, run it, and return the result."""

    plan = sqclient.generate_plan(
        query.query, query.index, util.get_schema(sqclient, query.index), examples=query.examples
    )
    _, result = sqclient.run_plan(plan)
    return QueryResult(plan=plan, result=result)
