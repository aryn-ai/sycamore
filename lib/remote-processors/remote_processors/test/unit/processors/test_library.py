from remote_processors.server.processor_registry import ProcessorRegistry


class TestProcessorLibrary:
    def test_that_processor_names_are_unique(self):
        """
        The ProcessorRegistry performs this validation
        on its own at construction time so just use it here
        """
        ProcessorRegistry()
