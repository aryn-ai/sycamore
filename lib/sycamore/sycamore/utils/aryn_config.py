import os
import pathlib
import yaml
import threading
from typing import Optional
import logging

_DEFAULT_PATH = os.path.join(pathlib.Path.home(), ".aryn", "config.yaml")

class ArynConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

_global_aryn_config_lock = threading.Lock()
_global_aryn_config: Optional[ArynConfig] = None

def get_aryn_api_key(requested_yaml_config_path: str = None) -> str:
    api_key = os.environ.get("ARYN_API_KEY")
    if api_key:
        return api_key
    
    global_aryn_config = get_aryn_config(requested_yaml_config_path)
    return global_aryn_config.__dict__.get("aryn_token", "")


def get_aryn_config(requested_yaml_config_path: str = None) -> ArynConfig:
    config_path = requested_yaml_config_path or os.environ.get("ARYN_CONFIG") or _DEFAULT_PATH
    global _global_aryn_config
    with _global_aryn_config_lock:
        if _global_aryn_config is None:
            config = ArynConfig()
            try:
                with open(config_path, "r") as yaml_file:
                    aryn_env = yaml.safe_load(yaml_file)
                    if aryn_env is None or not isinstance(aryn_env, dict):
                        logging.debug("Aryn YAML config appears to be empty.")
                        return
                    config.__dict__.update(aryn_env)
                    logging.debug(f"Aryn configuration: {vars(config)}")
            except FileNotFoundError as err:
                if config_path == _DEFAULT_PATH:
                    logging.debug(f"Aryn config YAML not present: {err}")
                else:
                    logging.error(f"Unable to load specified aryn config {config_path}: {err}")
            except yaml.scanner.ScannerError as err:
                logging.error(f"Unable to load {config_path}: {err}. Ignoring the config file.")
            _global_aryn_config = config
        
        return _global_aryn_config
    
