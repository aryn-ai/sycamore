from remote_processors.processors.debug_processor import DebugResponseProcessor
from remote_processors.test.utils import dummy_search_request, dummy_search_response


class TestDebugProcessor:

    def test_debug_processor_does_not_modify_search_response(self):
        req = dummy_search_request()
        res = dummy_search_response()
        debug_processor = DebugResponseProcessor.from_config({})
        processed = debug_processor.process_response(req, res)
        assert processed == res
