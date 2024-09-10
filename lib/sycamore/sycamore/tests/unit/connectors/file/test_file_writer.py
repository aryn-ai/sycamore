import sycamore
from sycamore.tests.config import TEST_DIR
from sycamore.connectors.file.file_writer import document_to_json_bytes

# To re-generate the input file, change the False to True, and in the root directory, run
# poetry run python lib/sycamore/sycamore/tests/unit/connectors/file/test_file_writer.py
if False:
    from sycamore.transforms.partition import ArynPartitioner

    (
        sycamore.init()
        .read.binary(paths="./lib/sycamore/sycamore/tests/resources/data/pdfs/Ray_page11.pdf", binary_format="pdf")
        .partition(ArynPartitioner(extract_images=True, use_partitioning_service=False, use_cache=False))
        .materialize(
            path="./lib/sycamore/sycamore/tests/resources/data/materialize/json_writer",
            source_mode=sycamore.MATERIALIZE_USE_STORED,
        )
        .execute()
    )


def test_json_bytes_with_bbox_image():
    docs = (
        sycamore.init(exec_mode=sycamore.ExecMode.LOCAL)
        .read.materialize(path=TEST_DIR / "resources/data/materialize/json_writer")
        .take_all()
    )
    # TODO: once we support writers in local mode, switch this to be
    # .write.json(tmpdir)
    # running as part of ray, it's too slow
    for d in docs:
        _ = document_to_json_bytes(d)
