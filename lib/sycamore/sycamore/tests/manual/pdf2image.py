import sys
import logging

from sycamore.utils.pdf import PdfToImageFiles


def main(args=None) -> int:
    if args is None:
        args = sys.argv[1:]
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname).1s %(asctime)s %(message)s",
        datefmt="%Y%m%d_%H%M%S",
    )
    fn = args.pop(0)
    full(fn)
    bail(fn)
    return 0


def full(fn: str) -> None:
    with PdfToImageFiles(pdf_path=fn, file_dir="/tmp") as gen:
        for p in gen:
            p.unlink()


def bail(fn: str) -> None:
    with PdfToImageFiles(pdf_path=fn, file_dir="/tmp") as gen:
        cnt = 0
        for p in gen:
            print(cnt, p, flush=True)
            p.unlink()
            cnt += 1
            if cnt > 20:
                break
        print("done loop", flush=True)
    print("after with", flush=True)


if __name__ == "__main__":
    sys.exit(main())
