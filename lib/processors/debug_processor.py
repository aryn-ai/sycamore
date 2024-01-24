
from lib.processors.processor import RequestProcessor, ResponseProcessor
from lib.search_request import SearchRequest
from lib.search_response import SearchResponse


class DebugResponseProcessor(ResponseProcessor):

    def from_config(configuration_chunk) -> ResponseProcessor:
        return DebugResponseProcessor()
    
    def get_class_name() -> str:
        return "debug-response"
        
    def process_response(search_request: SearchRequest, search_response: SearchResponse) -> SearchResponse:
        print(str(search_request))
        print(str(search_response))
        return search_response
    
class DebugRequestProcessor(RequestProcessor):

    def from_config(configuration_chunk) -> RequestProcessor:
        return DebugRequestProcessor()
    
    def get_class_name() -> str:
        return "debug-request"
    
    def process_request(search_request: SearchRequest) -> SearchRequest:
        print(str(search_request))
        return search_request
