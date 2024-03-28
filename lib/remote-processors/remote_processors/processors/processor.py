from abc import ABC, abstractmethod

from remote_processors import SearchRequest
from remote_processors import SearchResponse


class ResponseProcessor(ABC):
    @staticmethod
    @abstractmethod
    def from_config(configuration_chunk) -> "ResponseProcessor":
        raise NotImplementedError("abstract method `from_config` is not implemented")

    @abstractmethod
    def process_response(self, search_request: SearchRequest, search_response: SearchResponse) -> SearchResponse:
        raise NotImplementedError("abstract method `process_response` is not implemented")

    @staticmethod
    @abstractmethod
    def get_class_name() -> str:
        raise NotImplementedError("abstract method `get_class_name` is not implemented")


class RequestProcessor(ABC):
    @staticmethod
    @abstractmethod
    def from_config(configuration_chunk) -> "RequestProcessor":
        raise NotImplementedError("abstract method `from_config` is not implemented")

    @abstractmethod
    def process_request(self, search_request: SearchRequest) -> SearchRequest:
        raise NotImplementedError("abstract method `process_request` is not implemented")

    @staticmethod
    @abstractmethod
    def get_class_name() -> str:
        raise NotImplementedError("abstract static method `get_class_name` is not implemented")
