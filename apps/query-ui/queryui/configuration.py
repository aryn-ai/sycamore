from typing import Optional

from sycamore.query.client import SycamoreQueryClient

"""
If you want to use custom LLMs or a custom context to override opensearch parameters, edit the
get_sycamore_query_client function below.

For example, you can write:
    context = sycamore.init(params={
        "default": {"llm": OpenAI(OpenAIModels.GPT_4O.value, cache=cache_from_path(s3_cache_path))},
        "opensearch": {"os_client_args": get_opensearch_client_args(),
                       "text_embedder": SycamoreQueryClient.default_text_embedder()}
    })
    return SycamoreQueryClient(context=context, trace_dir=trace_dir)

Note that if you include keys in your configuration, it is safer to write:
    import mycompany_configuration
    return mycompany_configuration.get_sycamore_query_client(s3_cache_path, trace_dir)

We require you to edit this file so that if the arguments to get_sycamore_query_client change you
will get a merge conflict.
"""


def get_sycamore_query_client(
    s3_cache_path: Optional[str] = None, trace_dir: Optional[str] = None
) -> SycamoreQueryClient:
    return SycamoreQueryClient(s3_cache_path=s3_cache_path, trace_dir=trace_dir)
