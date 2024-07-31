from typing import Optional, Any


class Config:

    OPENSEARCH_CLIENT_CONFIG = "opensearch.client_config"
    OPENSEARCH_INDEX_NAME = "opensearch.index_name"
    OPENSEARCH_INDEX_SETTINGS = "opensearch.index_settings"

    LLM = "llm"

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        self.config: dict[str, Any] = config or dict()

    def get(self, key: str) -> Optional[Any]:
        return self.config.get(key, None)

    def set(self, key: str, val: Any) -> Any:
        self.config[key] = val
        return self.config[key]
