from dataclasses import dataclass
from typing import Optional, Any

from sycamore.llms import LLM


@dataclass
class Config:

    opensearch_client_config: Optional[dict[str, Any]] = None
    opensearch_index_name: Optional[str] = None
    opensearch_index_settings: Optional[dict[str, Any]] = None

    llm: Optional[LLM] = None
