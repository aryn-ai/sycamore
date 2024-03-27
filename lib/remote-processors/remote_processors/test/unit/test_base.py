from remote_processors.test.utils import dummy_search_request, dummy_search_response


class TestBase:
    def test_that_testing_works(self):
        assert True

    def test_that_utils_work(self):
        dummy_search_response()
        dummy_search_request()
