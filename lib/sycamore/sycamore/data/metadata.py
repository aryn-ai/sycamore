from sycamore.data import MetadataDocument
from sycamore.utils.thread_local import ThreadLocalAccess, ADD_METADATA_TO_OUTPUT


def add_metadata(**metadata):
    ThreadLocalAccess(ADD_METADATA_TO_OUTPUT).get().append(MetadataDocument(**metadata))


# At some point we should define particular forms of metadata like metrics
# Maybe following https://github.com/prometheus/OpenMetrics/blob/main/specification/OpenMetrics.md
# as a structure for the metrics -- too complex for now.
