import os
import pathlib
import yaml
import logging
import threading

_DEFAULT_PATH = os.path.join(pathlib.Path.home(), ".aryn", "config.yaml")


class ArynConfig:
    _global_aryn_config_lock = threading.Lock()
    _global_aryn_config = None

    @classmethod
    def get_aryn_api_key(cls, config_path: str = "") -> str:
        api_key = os.environ.get("ARYN_API_KEY")
        if api_key:
            return api_key

        with cls._global_aryn_config_lock:
            if cls._global_aryn_config:
                return cls._global_aryn_config.get("aryn_token", "")

        cls._get_aryn_config(config_path)

        with cls._global_aryn_config_lock:
            if cls._global_aryn_config:
                return cls._global_aryn_config.get("aryn_token", "")
            else:
                return ""

    @classmethod
    def _get_aryn_config(cls, config_path: str = "") -> None:
        with cls._global_aryn_config_lock:
            if cls._global_aryn_config:
                return
            config_path = config_path or os.environ.get("ARYN_CONFIG") or _DEFAULT_PATH

            try:
                with open(config_path, "r") as yaml_file:
                    aryn_config = yaml.safe_load(yaml_file)
                    cls._global_aryn_config = aryn_config
                    if not isinstance(aryn_config, dict):
                        logging.warning("Aryn YAML config appears to be empty.")
                        return
                    logging.debug(f"Aryn configuration: {aryn_config}")
            except FileNotFoundError as err:
                logging.error(f"Unable to load aryn config {config_path}: {err}")
            except yaml.scanner.ScannerError as err:
                logging.error(f"Unable to parse {config_path}: {err}. Ignoring the config file.")
