import grpc

import pytest
from remote_processors.response_processor_service_pb2 import ProcessResponseCall
from remote_processors.server.pipeline import BadPipelineConfigError
from remote_processors.server.remote_processor_service import RemoteProcessorService
from remote_processors.test.utils import dummy_search_request, dummy_search_response, TESTS_DIR


class TestRemoteProcessorService:
    def test_parse_valid_config(self):
        RemoteProcessorService(TESTS_DIR / "resources/configs/valid.yml")

    def test_parse_malformed_configs_then_fail(self):
        with pytest.raises(BadPipelineConfigError):
            RemoteProcessorService(TESTS_DIR / "resources/configs/malformed/not_a_list.yml")

        with pytest.raises(BadPipelineConfigError):
            RemoteProcessorService(TESTS_DIR / "resources/configs/malformed/pipeline_not_a_map.yml")

        with pytest.raises(BadPipelineConfigError):
            RemoteProcessorService(TESTS_DIR / "resources/configs/malformed/pipeline_with_many_keys.yml")

        with pytest.raises(BadPipelineConfigError):
            RemoteProcessorService(TESTS_DIR / "resources/configs/malformed/dupe_pipeline_names.yml")

    def test_process_response_pipeline_found(self):
        rps = RemoteProcessorService(TESTS_DIR / "resources/configs/valid.yml")
        prr = ProcessResponseCall(
            search_response=dummy_search_response(), search_request=dummy_search_request(), processor_name="debug"
        )
        response = rps.ProcessResponse(prr, None)
        assert response.search_response == dummy_search_response()

    def test_process_response_pipeline_not_found(self, mocker):
        class MockContext:
            def set_code(self, code):
                assert code == grpc.StatusCode.NOT_FOUND

            def set_details(self, message):
                pass

        context = MockContext()
        code_spy = mocker.spy(context, "set_code")
        detail_spy = mocker.spy(context, "set_details")
        rps = RemoteProcessorService(TESTS_DIR / "resources/configs/valid.yml")
        prr = ProcessResponseCall(
            search_response=dummy_search_response(), search_request=dummy_search_request(), processor_name="missing"
        )
        with pytest.raises(KeyError):
            rps.ProcessResponse(prr, context)
        assert code_spy.call_count == 1
        assert detail_spy.call_count == 1
