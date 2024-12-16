from typing import Optional

from sycamore import ExecMode
from sycamore.query.client import SycamoreQueryClient

"""
If you want to use custom LLMs or a custom context to override opensearch parameters, edit the
get_sycamore_query_client function below.

For example, you can write:
    context = sycamore.init(params={
        "default": {"llm": OpenAI(OpenAIModels.GPT_4O.value, cache=cache_from_path(llm_cache_dir))},
        "opensearch": {"os_client_args": get_opensearch_client_args(),
                       "text_embedder": SycamoreQueryClient.default_text_embedder()}
    })
    return SycamoreQueryClient(context=context, trace_dir=trace_dir)

Note that if you include keys in your configuration, it is safer to write:
    import mycompany_configuration
    return mycompany_configuration.get_sycamore_query_client(llm_cache_dir, trace_dir)

We require you to edit this file so that if the arguments to get_sycamore_query_client change you
will get a merge conflict.
"""


def get_sycamore_query_client(
    llm_cache_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    exec_mode: ExecMode = ExecMode.RAY,
) -> SycamoreQueryClient:
    return SycamoreQueryClient(llm_cache_dir=llm_cache_dir, cache_dir=cache_dir, sycamore_exec_mode=exec_mode)
