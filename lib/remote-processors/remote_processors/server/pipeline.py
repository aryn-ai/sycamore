from typing import Any, Union, List
from remote_processors.processors import RequestProcessor, ResponseProcessor
from remote_processors import SearchRequest, SearchResponse

from remote_processors.server.processor_registry import ProcessorRegistry

PROCESSOR_LIST_FIELD = "processors"


class Pipeline:
    """Class representing a sequence of processors. This is the unit that's served by the RPS."""

    def __init__(self, name: str, config: dict[str, Any], pr: ProcessorRegistry):
        """Create a processing pipeline

        Args:
            name (str): name of the pipeline. This corresponds to the "processor_name" field in a processing request
            config (dict[str, Any]): configuration of the pipeline. Which processors; their parameters
            pr (ProcessorRegistry): ref to registry object containing all processor classes
        """
        self._name = name
        self._processors = self._parse_config(config, pr)

    def _parse_config(
        self, config: dict[str, Any], pr: ProcessorRegistry
    ) -> Union[List[RequestProcessor], List[ResponseProcessor]]:
        """Parse a pipeline config section, such as

         .. code-block:: yaml
            pipeline:
              processors:
                  - debug-response:
                  - other-processor-that-isnt-implemented-yet:
                      param1: val1
                      param2: val2


        Args:
            config (dict[str, Any]): configuration to parse (has already been deserialized to dict)
            pr (ProcessorRegistry): the processor registry to look up processors from

        Raises:
            BadPipelineConfigError: unless `config` contains a non-empty list of correctly configured processors
            BadPipelineConfigError: if all the processors in the pipeline are not the same type (Request, Response)

        Returns:
            list[Union[ResponseProcessor, RequestProcessor]]: list of processors for this pipeline. must be homogenous in (RequestProcessor, ResponseProcessor)
        """
        processors = []
        if PROCESSOR_LIST_FIELD not in config:
            raise BadPipelineConfigError(f"Pipeline config for {self._name} missing {PROCESSOR_LIST_FIELD} field")
        proc_configs = config[PROCESSOR_LIST_FIELD]
        if not isinstance(proc_configs, list):
            raise BadPipelineConfigError(
                f"{PROCESSOR_LIST_FIELD} field for {self._name} must be a list of processor configurations"
            )
        if len(proc_configs) == 0:
            raise BadPipelineConfigError(f"Pipeline {self._name} must have at least one processor")
        # construct each processor in the pipeline
        for i, processor_cfg in enumerate(proc_configs):
            if not isinstance(processor_cfg, dict):
                raise BadPipelineConfigError(f"Configuration for processor {i} in pipeline {self._name} must be a map")
            if len(processor_cfg) != 1:
                raise BadPipelineConfigError(
                    f"Configuration for processor {i} must have exactly 1 key, the processor class name"
                )
            processor_type = list(processor_cfg.keys())[0]
            proc_clazz = pr.get_processor(processor_type)
            if proc_clazz is None:
                raise BadPipelineConfigError(f"Processor {processor_type} could not be found")
            proc = proc_clazz.from_config(processor_cfg[processor_type])
            processors.append(proc)
        if not (
            all([isinstance(p, RequestProcessor) for p in processors])
            or all([isinstance(p, ResponseProcessor) for p in processors])
        ):
            raise BadPipelineConfigError("All processors must be either request processors **or** response processors")
        return processors  # type: ignore

    def run_request_pipeline(self, search_request: SearchRequest) -> SearchRequest:
        """Runs a request pipeline (i.e. transforms requests pre-query)

        Args:
            search_request (SearchRequest): The request to process

        Raises:
            WrongPipelineError: if this is a response pipeline

        Returns:
            SearchRequest: The processed request
        """
        if any(isinstance(p, ResponseProcessor) for p in self._processors):
            raise WrongPipelineError(
                f"{self._name} contains response processors, so cannot be used as a request pipeline"
            )
        for proc in self._processors:
            search_request = proc.process_request(search_request)  # type: ignore
        return search_request

    def run_response_pipeline(self, search_request: SearchRequest, search_response: SearchResponse) -> SearchResponse:
        """Runs a response pipeline (i.e. transforms a search response post-query)

        Args:
            search_request (SearchRequest): The search request that created these results
            search_response (SearchResponse): The search results to process

        Raises:
            WrongPipelineError: if this is a request pipeline

        Returns:
            SearchResponse: The processed search results
        """
        if any(isinstance(p, RequestProcessor) for p in self._processors):
            raise WrongPipelineError(
                f"{self._name} contains request processors, so cannot be used as a response pipeline"
            )
        for proc in self._processors:
            search_response = proc.process_response(search_request, search_response)  # type: ignore
        return search_response


class BadPipelineConfigError(Exception):
    """Pipeline configuration is malformed in some wway"""

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self._msg = msg

    def __str__(self) -> str:
        return self._msg


class WrongPipelineError(Exception):
    """Pipeline is the wrong type (request vs response)"""

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self._msg = msg

    def __str__(self) -> str:
        return self._msg
