
from lib.processors.processor import RequestProcessor, ResponseProcessor
from lib.search_request import SearchRequest
from lib.search_response import SearchResponse

import logging
import sys

class DebugResponseProcessor(ResponseProcessor):

    def __init__(self):
        self._log = logging.getLogger("debug-processor")
        self._log.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)-5s][%(name)-12s] %(message)s'))
        self._log.addHandler(handler)


    def from_config(configuration_chunk) -> ResponseProcessor:
        return DebugResponseProcessor()
    
    def get_class_name() -> str:
        return "debug-response"
        
    def process_response(self, search_request: SearchRequest, search_response: SearchResponse) -> SearchResponse:
        self._log.info(f"search response: \n{search_response}")
        self._log.info(f"search request:  \n{search_request}")
        return search_response
    
class DebugRequestProcessor(RequestProcessor):

    def from_config(configuration_chunk) -> RequestProcessor:
        return DebugRequestProcessor()
    
    def get_class_name() -> str:
        return "debug-request"
    
    def process_request(self, search_request: SearchRequest) -> SearchRequest:
        print(str(search_request))
        return search_request
