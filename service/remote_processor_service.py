from concurrent import futures
from pathlib import Path
from typing import Any
import yaml
import grpc
from service.pipeline import BadPipelineConfigError, Pipeline
from service.processor_registry import ProcessorRegistry
from gen.response_processor_service_pb2_grpc import RemoteProcessorServiceServicer, add_RemoteProcessorServiceServicer_to_server
from gen.response_processor_service_pb2 import ProcessResponseRequest, ProcessResponseResponse

TP_MAX_WORKERS = 10

class RemoteProcessorService(RemoteProcessorServiceServicer):
    """Service driver for remote processing requests
    """

    def __init__(self, configuration_file: Path):
        """Constructorr. Parses configuration file to create served pipelines

        Args:
            configuration_file (Path): path to a yaml config file that contains all pipeline definitions for this instance
        """
        self._config_file = configuration_file
        self._pr = ProcessorRegistry()
        configuration = {}
        with open(configuration_file, "r") as f:
            configuration = yaml.safe_load(f)
        self._pipelines = self._parse_configuration(configuration)

    def _parse_configuration(self, configuration: list[dict[str, Any]]) -> dict[str, Pipeline]:
        """parse the config file and initialize the pipelines

        Args:
            configuration (list[dict[str, Any]]): service configuration (list of pipeline configs)

        Raises:
            BadPipelineConfigError: if configuration is not a list of valid pipeline configurations

        Returns:
            dict[str, Pipeline]: pipeline objects referenced by name
        """
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
    
    def ProcessResponse(self, request: ProcessResponseRequest, context):
        """Process a response. Entrypoint for ProcessResponse API

        Args:
            request (ProcessResponseRequest): The API request
            context (_type_): some grpc thing, idk

        Raises:
            KeyError: if the requested "processor_name" does not exist

        Returns:
            ProcessResponseResponse: the processed response
        """
        search_request = request.search_request
        search_response = request.search_response
        processor_name = request.processor_name
        if processor_name in self._pipelines:
            new_response = self._pipelines[processor_name].run_response_pipeline(search_request=search_request, search_response=search_response)
            return ProcessResponseResponse(search_response=new_response)
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f'Processor {processor_name} does not exist!')
            raise KeyError(f"Processor {processor_name} does not exist! Check your config file {self._config_file}")
        
    def start(self):
        """Start the server on port 2796 (ARYN on a keypad)

        Returns:
            Server: a grpc server object
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=TP_MAX_WORKERS))
        add_RemoteProcessorServiceServicer_to_server(self, server)
        server.add_insecure_port("[::]:2796")
        server.start()
        return server
        
    