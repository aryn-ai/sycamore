from typing import Dict, Optional


class Config:
    def __init__(self, config: Dict[str, str] = None) -> None:
        super().__init__()
        self.config: Dict[str, str] = config or dict()

    def get(self, key: str) -> str:
        return self.config.get(key)

    def set(self, key: str, val: str) -> str:
        self.config[key] = val
        return self.config[key]

    @property
    def openai_model_name(self) -> Optional[str]:
        return self.config.get("openai_model_name")

    @openai_model_name.setter
    def openai_model_name(self, value: str):
        self.config["openai_model_name"] = value
