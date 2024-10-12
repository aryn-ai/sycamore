import logging
from typing import Any, Dict, Optional, Set, Tuple

import ray

from sycamore.executor import _ray_logging_setup
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan


def ray_init(**ray_args):
    if ray.is_initialized():
        return

    if "logging_level" not in ray_args:
        ray_args.update({"logging_level": logging.INFO})
    if "runtime_env" not in ray_args:
        ray_args["runtime_env"] = {}
    if "worker_process_setup_hook" not in ray_args["runtime_env"]:
        ray_args["runtime_env"]["worker_process_setup_hook"] = _ray_logging_setup
    ray.init(**ray_args)


def get_schema(_client: SycamoreQueryClient, index: str) -> Dict[str, Tuple[str, Set[str]]]:
    return _client.get_opensearch_schema(index)


def generate_plan(_client: SycamoreQueryClient, query: str, index: str, examples: Optional[Any] = None) -> LogicalPlan:
    return _client.generate_plan(query, index, get_schema(_client, index), examples=examples)


def run_plan(_client: SycamoreQueryClient, plan: LogicalPlan) -> Tuple[str, Any]:
    return _client.run_plan(plan)


def get_opensearch_indices() -> Set[str]:
    return {x for x in SycamoreQueryClient().get_opensearch_indices() if not x.startswith(".")}
