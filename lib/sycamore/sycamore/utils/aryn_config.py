import os
import pathlib
import yaml
import logging

from typing import Dict, Any

DEFAULT_PATH = os.path.join(pathlib.Path.home(), ".aryn", "config.yaml")
_DEFAULT_PATH = DEFAULT_PATH
## ToDO: remove after 31/08/2024


class ArynConfig:
    @classmethod
    def get_aryn_api_key(cls, config_path: str = "") -> str:
        api_key = os.environ.get("ARYN_API_KEY")
        if api_key:
            return api_key

        return cls._get_aryn_config(config_path).get("aryn_token", "")

    @classmethod
    def _get_aryn_config(cls, config_path: str = "") -> Dict[Any, Any]:
        config_path = config_path or os.environ.get("ARYN_CONFIG") or _DEFAULT_PATH

        try:
            with open(config_path, "r") as yaml_file:
                aryn_config = yaml.safe_load(yaml_file)
                if not isinstance(aryn_config, dict):
                    logging.warning("Aryn YAML config appears to be empty.")
                    return {}
                logging.debug(f"Aryn configuration: {aryn_config}")
                return aryn_config
        except FileNotFoundError as err:
            if config_path == _DEFAULT_PATH:
                logging.debug(f"Unable to load Aryn config {config_path}: {err}")
            else:
                logging.error(f"Unable to load Aryn config {config_path}: {err}")
        except yaml.scanner.ScannerError as err:
            logging.error(f"Unable to parse {config_path}: {err}. Ignoring the config file.")

        return {}
