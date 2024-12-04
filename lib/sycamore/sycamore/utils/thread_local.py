import logging
import threading

logger = logging.getLogger(__name__)
# Used to inject metadata into the docset output stream to make it easier to use with
# transforms like map.
ADD_METADATA_TO_OUTPUT = "add_metadata_to_output"

# Create a thread-local data object
thread_local_data = threading.local()


class ThreadLocalAccess:
    def __init__(self, var_name):
        self.var_name = var_name

    def present(self):
        return hasattr(thread_local_data, self.var_name)

    def get(self):
        assert hasattr(thread_local_data, self.var_name), f"{self.var_name} not present in TLS"
        return getattr(thread_local_data, self.var_name)

    def set(self, value):
        assert hasattr(thread_local_data, self.var_name), f"{self.var_name} not present in TLS"
        setattr(thread_local_data, self.var_name, value)


class ThreadLocal(ThreadLocalAccess):
    def __init__(self, var_name, var_value):
        self.var_name = var_name
        self.var_value = var_value

    def __enter__(self):
        assert not hasattr(thread_local_data, self.var_name), f"{self.var_name} already set in TLS"
        setattr(thread_local_data, self.var_name, self.var_value)
        logger.debug(f"Thread-local variable '{self.var_name}' removed.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert hasattr(thread_local_data, self.var_name), f"{self.var_name} vanished from TLS"
        delattr(thread_local_data, self.var_name)
        logger.debug(f"Thread-local variable '{self.var_name}' removed.")
