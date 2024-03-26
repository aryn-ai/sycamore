
from lib.processors.processor import RequestProcessor, ResponseProcessor
from lib.search_request import SearchRequest
from lib.search_response import SearchResponse

import logging
import sys
import cbor2

_LOG = logging.getLogger("debug-processor")
_LOG.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)-5s][%(name)-12s] %(message)s'))
_LOG.addHandler(handler)

class DebugResponseProcessor(ResponseProcessor):

    def __init__(self, prefix: str = None):
        self._prefix = prefix

    def from_config(configuration_chunk) -> ResponseProcessor:
        if isinstance(configuration_chunk, dict) and "prefix" in configuration_chunk:
            return DebugResponseProcessor(configuration_chunk['prefix'])
        else:
            return DebugResponseProcessor()
    
    def get_class_name() -> str:
        return "debug-response"
        
    def process_response(self, search_request: SearchRequest, search_response: SearchResponse) -> SearchResponse:
        if self._prefix is not None:
            _LOG.info(self._prefix)
        _LOG.info(f"search response: \n{search_response}")
        _LOG.info(f"search request:  \n{search_request}")
        return search_response
    
class DebugRequestProcessor(RequestProcessor):

    def __init__(self): 
        pass

    def from_config(configuration_chunk) -> RequestProcessor:
        return DebugRequestProcessor()
    
    def get_class_name() -> str:
        return "debug-request"
    
    def process_request(self, search_request: SearchRequest) -> SearchRequest:
        _LOG.info(f"search request:  \n{search_request}")
        return search_request
