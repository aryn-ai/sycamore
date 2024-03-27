import pytest
from remote_processors.processors.debug_processor import DebugRequestProcessor, DebugResponseProcessor
from remote_processors.server.pipeline import PROCESSOR_LIST_FIELD, BadPipelineConfigError, Pipeline, WrongPipelineError
from remote_processors.server.processor_registry import ProcessorRegistry
from remote_processors.test.utils import dummy_search_request, dummy_search_response


class TestPipeline:

    pr = ProcessorRegistry()

    def test_valid_configuration(self):
        cfg = {
            PROCESSOR_LIST_FIELD: [
                {DebugResponseProcessor.get_class_name(): {"prefix": "prefix"}},
                {DebugResponseProcessor.get_class_name(): None},
            ]
        }
        pipeline = Pipeline("valid", cfg, self.pr)
        assert (
            pipeline.run_response_pipeline(dummy_search_request(), dummy_search_response()) == dummy_search_response()
        )

    def test_invalid_configurations(self):
        cfg = {"processors list": [{DebugResponseProcessor.get_class_name(): None}]}
        with pytest.raises(BadPipelineConfigError):
            Pipeline("invalid", cfg, self.pr)

        cfg = {PROCESSOR_LIST_FIELD: {DebugResponseProcessor.get_class_name(): None}}
        with pytest.raises(BadPipelineConfigError):
            Pipeline("invalid", cfg, self.pr)

        cfg = {PROCESSOR_LIST_FIELD: []}
        with pytest.raises(BadPipelineConfigError):
            Pipeline("invalid", cfg, self.pr)

        cfg = {PROCESSOR_LIST_FIELD: [DebugResponseProcessor.get_class_name()]}
        with pytest.raises(BadPipelineConfigError):
            Pipeline("invalid", cfg, self.pr)

        cfg = {
            PROCESSOR_LIST_FIELD: [
                {DebugResponseProcessor.get_class_name(): None, DebugRequestProcessor.get_class_name(): None}
            ]
        }
        with pytest.raises(BadPipelineConfigError):
            Pipeline("invalid", cfg, self.pr)

        cfg = {PROCESSOR_LIST_FIELD: [{"nonexistant_processor": None}]}
        with pytest.raises(BadPipelineConfigError):
            Pipeline("invalid", cfg, self.pr)

        cfg = {
            PROCESSOR_LIST_FIELD: [
                {DebugResponseProcessor.get_class_name(): None},
                {DebugRequestProcessor.get_class_name(): None},
            ]
        }
        with pytest.raises(BadPipelineConfigError):
            Pipeline("invalid", cfg, self.pr)

    def test_wrong_pipeline_type(self):
        res_cfg = {PROCESSOR_LIST_FIELD: [{DebugResponseProcessor.get_class_name(): None}]}
        response_pipeline = Pipeline("response", res_cfg, self.pr)
        req_cfg = {PROCESSOR_LIST_FIELD: [{DebugRequestProcessor.get_class_name(): None}]}
        request_pipeline = Pipeline("request", req_cfg, self.pr)
        with pytest.raises(WrongPipelineError):
            response_pipeline.run_request_pipeline(dummy_search_request())
        with pytest.raises(WrongPipelineError):
            request_pipeline.run_response_pipeline(dummy_search_request(), dummy_search_response())

    def test_pipeline_hits_all_processors(self, mocker):
        cfg = {
            PROCESSOR_LIST_FIELD: [
                {DebugResponseProcessor.get_class_name(): {"prefix": "prefix"}},
                {DebugResponseProcessor.get_class_name(): None},
            ]
        }
        pipeline = Pipeline("valid", cfg, self.pr)
        processor1 = pipeline._processors[0]
        processor2 = pipeline._processors[1]
        spy1 = mocker.spy(processor1, "process_response")
        spy2 = mocker.spy(processor2, "process_response")
        pipeline.run_response_pipeline(dummy_search_request(), dummy_search_response())
        assert spy1.call_count == 1
        assert spy2.call_count == 1
