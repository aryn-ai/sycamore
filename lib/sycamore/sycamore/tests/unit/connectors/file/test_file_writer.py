import glob
import json
import tempfile
from datetime import date, datetime

import sycamore
from sycamore.data import Document
from sycamore.tests.config import TEST_DIR

# To re-generate the input file, change the False to True, and in the root directory, run
# poetry run python lib/sycamore/sycamore/tests/unit/connectors/file/test_file_writer.py
if False:
    from sycamore.transforms.partition import ArynPartitioner

    (
        sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
        .read.binary(paths="./lib/sycamore/sycamore/tests/resources/data/pdfs/Ray_page11.pdf", binary_format="pdf")
        .partition(ArynPartitioner(extract_images=True, use_partitioning_service=False, use_cache=False))
        .materialize("./lib/sycamore/sycamore/tests/resources/data/materialize/json_writer")
        .execute()
    )


def impl_test_json_bytes_with_bbox_image(exec_mode):
    with tempfile.TemporaryDirectory() as tempdir:
        (
            sycamore.init(exec_mode=exec_mode)
            .read.materialize(path=TEST_DIR / "resources/data/materialize/json_writer")
            .write.json(tempdir)
        )

        nfiles = 0
        for name in glob.glob(f"{tempdir}/*"):
            with open(name, "r") as f:
                data = f.read()
                if len(data) == 0:  # ray test ends up with an empty file
                    continue
                nfiles = nfiles + 1
                _ = json.loads(data)

        assert nfiles == 1


def test_json_bytes_with_bbox_image():
    impl_test_json_bytes_with_bbox_image(sycamore.EXEC_LOCAL)


def test_doc_with_dates(exec_mode):
    with tempfile.TemporaryDirectory() as tempdir:
        doc = Document()
        doc.properties["d"] = datetime.now()
        try:
            sycamore.init(exec_mode=exec_mode).read.document([doc]).write.json(path=tempdir)
            assert True
        except TypeError:
            assert False
