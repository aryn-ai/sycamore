from os import PathLike
import os
from pathlib import Path
from typing import Optional
import yaml
import logging

_DEFAULT_PATH = Path.home() / ".aryn" / "config.yaml"
_ARYN_API_KEY_ENV_VAR = "ARYN_API_KEY"


class ArynConfig:
    def __init__(self, aryn_config_path: PathLike = _DEFAULT_PATH, aryn_api_key: Optional[str] = None):
        self._aryn_config_path = Path(aryn_config_path)
        self._aryn_api_key = aryn_api_key

    def api_key(self) -> str:
        if self._aryn_api_key is not None:
            return self._aryn_api_key
        if _ARYN_API_KEY_ENV_VAR in os.environ:
            return os.environ[_ARYN_API_KEY_ENV_VAR]
        if Path(self._aryn_config_path).exists():
            with open(self._aryn_config_path, "r") as f:
                data = yaml.safe_load(f)
                if "aryn_token" in data:
                    return data["aryn_token"]
        logging.warn(
            f"Could not find an aryn api key. Checked the {_ARYN_API_KEY_ENV_VAR} env "
            f"var, the {self._aryn_config_path} config file, and the aryn_api_key parameter"
        )
        return ""
