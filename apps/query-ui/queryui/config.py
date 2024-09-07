from sycamore.context import Context

from sycamore.query.client import SycamoreQueryClient

import logging

logging.basicConfig(level=logging.INFO)


"""
If you want to use custom LLMs (or just a custom context in general, you can create a file called custom_client.py
and overload 
    code:
        def get_sycamore_query_client(llm, s3_cache_path, os_client_args, trace_dir, **kwargs) -> Context:
"""


def get_default_sycamore_query_client(
    llm=None, s3_cache_path=None, os_client_args=None, trace_dir=None, **kwargs
) -> Context:
    return SycamoreQueryClient(s3_cache_path=s3_cache_path, os_client_args=os_client_args, trace_dir=trace_dir)

# this is for mypy
get_sycamore_query_client = None
try:
    from custom_client import get_sycamore_query_client

    logging.info("Using custom `get_sycamore_query_client(..)` from custom_client")
except (ImportError, AttributeError):
    logging.info("Falling back to default `get_sycamore_query_client(..)`")
    get_sycamore_query_client = get_default_sycamore_query_client
