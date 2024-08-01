from dataclasses import dataclass
from typing import Optional, Any

from sycamore.llms import LLM


@dataclass
class Config:
    """
    Context level configs accessible to transforms. Allows you to specify configurations once and reuse them
    across multiple invocations.


    Example:
     .. code-block:: python

        openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
        context = sycamore.init(
            config=Config(
                opensearch_index_name="toyindex",
                llm=openai_llm,
            )
        )


    """

    opensearch_client_config: Optional[dict[str, Any]] = None
    opensearch_index_name: Optional[str] = None
    opensearch_index_settings: Optional[dict[str, Any]] = None

    llm: Optional[LLM] = None
