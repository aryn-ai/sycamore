import sycamore
from sycamore.tests.unit.connectors.file.test_file_writer import impl_test_json_bytes_with_bbox_image


def test_json_bytes_with_bbox_image():
    impl_test_json_bytes_with_bbox_image(sycamore.EXEC_RAY)
