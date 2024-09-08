from sycamore.context import Context

from sycamore.query.client import SycamoreQueryClient

import logging

logging.basicConfig(level=logging.INFO)


"""
If you want to use custom LLMs (or just a custom context in general, you can create a file called custom_client.py
and overload 
    code:
        def get_custom_sycamore_query_client(llm, s3_cache_path, os_client_args, trace_dir, **kwargs) -> Context:
"""


def get_default_sycamore_query_client(
    llm=None, s3_cache_path=None, os_client_args=None, trace_dir=None, **kwargs
) -> Context:
    return SycamoreQueryClient(s3_cache_path=s3_cache_path, os_client_args=os_client_args, trace_dir=trace_dir)


def get_sycamore_query_client(**kwargs) -> Context:
    try:
        import custom_client

        get_sycamore_query_client_impl = custom_client.get_custom_sycamore_query_client
        logging.info("Using custom `get_custom_sycamore_query_client(..)` from custom_client")
    except (ImportError, AttributeError):
        logging.info("Using default `get_sycamore_query_client(..)`")
        get_sycamore_query_client_impl = get_default_sycamore_query_client

    return get_sycamore_query_client_impl(**kwargs)
