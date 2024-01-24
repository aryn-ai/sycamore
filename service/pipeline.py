from typing import Any, Union
from lib.processors.processor import RequestProcessor, ResponseProcessor
from lib.search_request import SearchRequest
from lib.search_response import SearchResponse

from service.processor_registry import ProcessorRegistry

PROCESSOR_LIST_FIELD = "processors"

class Pipeline:

    def __init__(self, name: str, config: dict[str, Any], pr: ProcessorRegistry):
        self._name = name
        self._processors = self._parse_config(config, pr)

    def _parse_config(self, config: dict[str, Any], pr: ProcessorRegistry) -> list[Union[ResponseProcessor, RequestProcessor]]:
        processors = []
        if PROCESSOR_LIST_FIELD not in config:
            raise BadPipelineConfigError(f"Pipeline config for {self._name} missing {PROCESSOR_LIST_FIELD} field")
        proc_configs = config[PROCESSOR_LIST_FIELD]
        if not isinstance(proc_configs, list):
            raise BadPipelineConfigError(f"{PROCESSOR_LIST_FIELD} field for {self._name} must be a list of processor configurations")
        if len(proc_configs) == 0:
            raise BadPipelineConfigError(f"Pipeline {self._name} must have at least one processor")
        for i, processor_cfg in enumerate(proc_configs):
            if not isinstance(processor_cfg, dict):
                raise BadPipelineConfigError(f"Configuration for processor {i} in pipeline {self._name} must be a map")
            if len(processor_cfg) != 1:
                raise BadPipelineConfigError(f"Configuration for processor {i} must have exactly 1 key, the processor class name")
            processor_type = list(processor_cfg.keys())[0]
            proc = pr.get_processor(processor_type).from_config(processor_cfg[processor_type])
            processors.append(proc)
        if not (all([isinstance(p, RequestProcessor) for p in processors]) or \
                all([isinstance(p, ResponseProcessor) for p in processors])):
            raise BadPipelineConfigError(f"All processors must be either request processors **or** response processors")
        return processors
    
    def run_request_pipeline(self, search_request: SearchRequest) -> SearchRequest:
        if isinstance(self._processors[0], ResponseProcessor):
            raise WrongPipelineError(f"{self._name} is a response pipeline, not a request pipeline")
        for proc in self._processors:
            search_request = proc.process_request(search_request)
        return search_request
    
    def run_response_pipeline(self, search_request: SearchRequest, search_response: SearchResponse) -> SearchResponse:
        if isinstance(self._processors[0], RequestProcessor):
            raise WrongPipelineError(f"{self._name} is a request pipeline, not a response pipeline")
        for proc in self._processors:
            search_response = proc.process_response(search_request, search_response)
        return search_response

        

class BadPipelineConfigError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self._msg = msg

    def __str__(self) -> str:
        return self._msg
    
class WrongPipelineError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self._msg = msg

    def __str__(self) -> str:
        return self._msg