from pathlib import Path
from typing import Any
import yaml
from service.pipeline import BadPipelineConfigError, Pipeline

from service.processor_registry import ProcessorRegistry

class RemoteProcessorService:

    def __init__(self, configuration_file: Path):
        self._config_file = configuration_file
        self._pr = ProcessorRegistry()
        configuration = {}
        with open(configuration_file, "r") as f:
            configuration = yaml.safe_load(f)
        self._pipelines = self._parse_configuration(configuration)

    def _parse_configuration(self, configuration: list[dict[str, Any]]) -> dict[str, Pipeline]:
        if not isinstance(configuration, list):
            raise BadPipelineConfigError("Config file must be a list of pipeline configurations")
        pipelines = {}
        for i, cfg in enumerate(configuration):
            if not isinstance(cfg, dict):
                raise BadPipelineConfigError(f"Pipeline {i} must be a map")
            if len(cfg) != 1:
                raise BadPipelineConfigError(f"Pipeline {i} must have exactly one key, the pipeline name")
            name = list(cfg.keys())[0]
            if name in pipelines:
                raise BadPipelineConfigError(f"Pipeline names must be unique. Found duplicate: {name}")
            pipeline = Pipeline(name, cfg[name], self._pr)
            pipelines[name] = pipeline
        return pipelines
        
    