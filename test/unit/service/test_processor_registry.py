import pytest
import gc
from proto_remote_processor.search_request_pb2 import SearchRequest
from proto_remote_processor.search_response_pb2 import SearchResponse
from lib.processors.debug_processor import DebugRequestProcessor, DebugResponseProcessor
from lib.processors.processor import ResponseProcessor
from service.processor_registry import ProcessorRegistry, DuplicatedProcessorNameError


class TestProcessorRegistry:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        print("cleaning")
        gc.collect()

    def test_processor_registry_has_no_duplicate_names(self):
        """Processor Registry tests this itself
        """
        pr = ProcessorRegistry()

    def test_processor_registry_with_duplicated_names_fails(self):
        """Construct some processor subclasses that break this
        """
        class Proc1(ResponseProcessor):
            @staticmethod
            def from_config(configuration_chunk) -> "ResponseProcessor":
                raise NotImplementedError("abstract method `from_config` is not implemented")
            
            def process_response(self, search_request: SearchRequest, search_response: SearchResponse) -> SearchResponse:
                raise NotImplementedError("abstract method `process_response` is not implemented")
            
            @staticmethod
            def get_class_name() -> str:
                return "dupe_name"
        
        class Proc2(ResponseProcessor):
            @staticmethod
            def from_config(configuration_chunk) -> "ResponseProcessor":
                raise NotImplementedError("abstract method `from_config` is not implemented")
            
            def process_response(self, search_request: SearchRequest, search_response: SearchResponse) -> SearchResponse:
                raise NotImplementedError("abstract method `process_response` is not implemented")
            
            @staticmethod
            def get_class_name() -> str:
                return "dupe_name"
        
        with pytest.raises(DuplicatedProcessorNameError) as e_info:
            pr = ProcessorRegistry()

        assert str(e_info.value) == "Duplicated processor names: \ndupe_name:\n\tProc1\n\tProc2"

    def test_processor_registry_isnt_zonked_forever_now(self):
        """Make sure Proc1 and Proc2 aren't forever lodged in the runtime
        """
        pr = ProcessorRegistry()

    def test_processor_registry_contains_known_processors(self):
        pr = ProcessorRegistry()
        assert pr.get_processor(DebugRequestProcessor.get_class_name()) is DebugRequestProcessor
        assert pr.get_processor(DebugResponseProcessor.get_class_name()) is DebugResponseProcessor