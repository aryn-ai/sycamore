# This is a FastAPI server that wraps Sycamore Query in a REST service.
#
# Run with:
#   poetry run fastapi dev queryserver/main.py

import asyncio
import logging
import os
import tempfile
import time
from typing import Annotated, Any, List, Optional

from fastapi import FastAPI, Path
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from sycamore import DocSet
from sycamore.data import Document, MetadataDocument
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.schema import OpenSearchSchema

logger = logging.getLogger("uvicorn.error")


app = FastAPI()

# The query and LLM cache paths.
CACHE_PATH = os.getenv("QUERYSERVER_CACHE_PATH", os.path.join(tempfile.gettempdir(), "queryserver_cache"))
LLM_CACHE_PATH = os.getenv("QUERYSERVER_LLM_CACHE_PATH", os.path.join(tempfile.gettempdir(), "queryserver_llm_cache"))

sqclient = SycamoreQueryClient(llm_cache_dir=LLM_CACHE_PATH, cache_dir=CACHE_PATH)


class Index(BaseModel):
    """Represents an index that can be queried."""

    index: str
    """The index name."""

    description: Optional[str] = None
    """Description of the index."""

    index_schema: OpenSearchSchema
    """The schema for this index."""


class Query(BaseModel):
    """Query an index with a given natural language query string. One of 'query' or 'plan' must be provided."""

    index: str
    """The index to query."""

    query: Optional[str] = None
    """The natural language query to run. if specified, `plan` must not be set."""

    plan: Optional[LogicalPlan] = None
    """The logical query plan to run. If specified, `query` must not be set."""

    stream: bool = False
    """If true, query results will be streamed back to the client as they are generated."""


class QueryResult(BaseModel):
    """The result of a non-streaming query."""

    query_id: str
    """The unique ID of the query operation."""

    plan: LogicalPlan
    """The logical query plan that was executed."""

    result: Any
    """The result of the query operation. Depending on the query, this could be a list of documents,
    a single document, a string, an integer, etc.
    """

    retrieved_docs: List[str]
    """A list of document paths for the documents retrieved by the query."""


@app.get("/v1/indices")
async def list_indices() -> List[Index]:
    """List all available indices."""

    retval = []
    # Exclude the 'internal' indices that start with a dot.
    indices = {x for x in sqclient.get_opensearch_indices() if not x.startswith(".")}
    for index in indices:
        index_schema = sqclient.get_opensearch_schema(index)
        retval.append(Index(index=index, index_schema=index_schema))
    return retval


@app.get("/v1/index/{index}")
async def get_index(
    index: Annotated[str, Path(title="The ID of the index to get")],
) -> Index:
    """Return details on the given index."""

    schema = sqclient.get_opensearch_schema(index)
    return Index(index=index, index_schema=schema)


@app.post("/v1/plan")
async def generate_plan(query: Query) -> LogicalPlan:
    """Generate a query plan for the given query, but do not run it."""

    if query.query is None:
        raise ValueError("query is required")
    if query.plan is not None:
        raise ValueError("plan must not be specified")

    plan = sqclient.generate_plan(query.query, query.index, sqclient.get_opensearch_schema(query.index))
    return plan


def doc_to_json(doc: Document) -> Optional[dict[str, Any]]:
    """Render a Document as a JSON object. Returns None for MetadataDocuments."""
    NUM_TEXT_CHARS_GENERATE = 1024

    if isinstance(doc, MetadataDocument):
        return None

    props_dict = {}
    props_dict.update(doc.properties)
    if "_schema" in props_dict:
        del props_dict["_schema"]
    if "_schema_class" in props_dict:
        del props_dict["_schema_class"]
    if "_doc_source" in props_dict:
        del props_dict["_doc_source"]
    props_dict["text_representation"] = (
        doc.text_representation[:NUM_TEXT_CHARS_GENERATE] if doc.text_representation is not None else None
    )
    return props_dict



async def run_query_stream(query: Query) -> EventSourceResponse:
    """Streaming version of run_query. Returns a stream of results as they are generated."""

    async def query_runner():
        try:
            logger.info(f"Generating plan for {query.index}: {query.query}")
            yield {
                "event": "status",
                "data": "Generating plan",
            }
            await asyncio.sleep(0.1)
            plan = sqclient.generate_plan(query.query, query.index, sqclient.get_opensearch_schema(query.index))
            logger.info(f"Generated plan: {plan}")
            # Don't want to return these through the API.
            plan.llm_plan = None
            plan.llm_prompt = None
            yield {
                "event": "plan",
                "data": plan.model_dump_json(),
            }
            await asyncio.sleep(0.1)
            logger.info("Running plan")
            yield {
                "event": "status",
                "data": "Running plan",
            }
            await asyncio.sleep(0.1)
            sqresult = sqclient.run_plan(plan)
            t1 = time.time()
            num_results = 0
            if isinstance(sqresult.result, DocSet):
                logger.info("Got DocSet result")
                for doc in sqresult.result.take_all():
                    rendered = doc_to_json(doc)
                    logger.info(f"Doc: {rendered}")
                    if rendered is not None:
                        num_results += 1
                        yield {
                            "event": "result_doc",
                            "data": rendered,
                        }
                        await asyncio.sleep(0.1)
            else:
                num_results += 1
                yield {
                    "event": "result",
                    "data": sqresult.result,
                }
                await asyncio.sleep(0.1)

            for doc in sqresult.retrieved_docs():
                yield {
                    "event": "retrieved_doc",
                    "data": doc,
                }
                await asyncio.sleep(0.1)

            t2 = time.time()
            logger.info(f"Finished query in {t2 - t1:.2f} seconds with {num_results} results")
            yield {
                "event": "status",
                "data": f"Query complete - {num_results} results in {t2 - t1:.2f} seconds",
            }
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Disconnected from client")

    return EventSourceResponse(query_runner())


@app.post("/v1/query", response_model=None)
async def run_query(query: Query) -> EventSourceResponse | QueryResult:
    """Run the given query.

    If the `stream` parameter is set to true, the result will be streamed back to the client as a series of SSE events.
    Otherwise, the result will be returned as a QueryResult object.
    """

    logger.info(f"Running query: {query}")

    if query.query is None and query.plan is None:
        raise ValueError("query or plan is required")
    if query.query is not None and query.plan is not None:
        raise ValueError("query and plan cannot both be specified")

    if query.stream:
        return await run_query_stream(query)

    if query.plan is None:
        assert query.query is not None
        logger.info(f"Generating plan for {query.index}: {query.query}")
        plan = sqclient.generate_plan(query.query, query.index, sqclient.get_opensearch_schema(query.index))
        logger.info(f"Generated plan: {plan}")
    else:
        plan = query.plan

    sqresult = sqclient.run_plan(plan)
    returned_plan = sqresult.plan

    # Don't want to return these through the API.
    returned_plan.llm_plan = None
    returned_plan.llm_prompt = None

    query_result = QueryResult(query_id=sqresult.query_id, plan=returned_plan, result=[], retrieved_docs=[])

    if isinstance(sqresult.result, DocSet):
        logger.info("Got DocSet result")
        for doc in sqresult.result.take_all():
            rendered = doc_to_json(doc)
            logger.info(f"Doc: {rendered}")
            if rendered is not None:
                query_result.result.append(rendered)
    else:
        query_result.result = sqresult.result

    query_result.retrieved_docs = sqresult.retrieved_docs()
    return query_result
