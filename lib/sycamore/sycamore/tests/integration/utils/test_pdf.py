import os
import sys
import logging
from tempfile import TemporaryDirectory

from sycamore.tests.config import TEST_DIR
from sycamore.utils.pdf import PdfToImageFiles


def test_basic():
    fn = TEST_DIR / "resources/data/pdfs/Ray.pdf"
    with TemporaryDirectory(prefix="test_pdf") as dir:
        n = full(fn, dir)
        assert n == 17
        bail(fn, dir, 8)
        with os.scandir(dir) as scan:
            for ent in scan:
                assert False, "temp dir should be empty now"


def full(fn: str, dir: str) -> int:
    cnt = 0
    with PdfToImageFiles(pdf_path=fn, file_dir=dir) as gen:
        for p in gen:
            p.unlink()
            cnt += 1
    return cnt


def bail(fn: str, dir: str, limit: int) -> None:
    with PdfToImageFiles(pdf_path=fn, file_dir=dir) as gen:
        cnt = 0
        for p in gen:
            logging.info(f"{cnt} {p}")
            p.unlink()
            cnt += 1
            if cnt > limit:
                break
        logging.info("done loop")
    logging.info("after with")


def main(args=None) -> int:
    if args is None:
        args = sys.argv[1:]
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname).1s %(asctime)s %(message)s",
        datefmt="%Y%m%d_%H%M%S",
    )
    fn = args.pop(0)
    with TemporaryDirectory(prefix="test_pdf") as dir:
        n = full(fn, dir)
        bail(fn, dir, n // 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
