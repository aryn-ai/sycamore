import sys
import logging
import time
import queue
import datetime
import multiprocessing
import tempfile

from pathlib import Path
from threading import Thread

from sycamore.utils.pdf import pdf_to_image_files
from sycamore.utils.time_trace import LogTime


def main():
    logging.basicConfig(level=logging.INFO)
    threads = []
    assert len(sys.argv) > 1
    for i, f in enumerate(sys.argv[1:]):
        print(f"Thread for {f}")
        t = Thread(target=process_file, args=[f, i])
        t.start()
        threads.append(t)

    while True:
        time.sleep(1)
        LogTime("snapshot", point=True)


def process_file(name, num):
    logging.info(f"Thread for {name}/{num}")
    n = 0
    q = queue.Queue(4)
    t = Thread(target=consume_q, args=[q])
    t.start()
    with tempfile.TemporaryDirectory(prefix="/dev/shm/pdf2image_memusage-") as tempdirname:
        logging.info(f"temporary files for {name}/{num} in {tempdirname}")
        while True:
            start = datetime.datetime.now()

            # streamed_batched:
            #   ~2.75s per iteration for transformer paper
            # PdfToImageFilesStreamed (on branch, uses multiprocessing for execution):
            #   spawn is ~4s/iteration for transformer paper
            #   forkserver is also ~4s/iteration
            #   fork is 2.75s/iteration

            # pdf_to_image hacked to emit:
            #     PIL.Images grows a lot and was still growing at 10min
            #     Bytes (raw ppm) is about 2x smaller than PIL.Images
            #     Files is about 1.5x smaller than bytes

            # for i in convert_from_path_streamed_batched(name, batch_size=1):
            for i in pdf_to_image_files(name, Path(tempdirname)):
                q.put(i)
                pass
            n = n + 1
            elapsed = datetime.datetime.now() - start

            logging.info(f"Completed pass {n} on {name} in {elapsed}")


def consume_q(q):
    while True:
        i = q.get()
        time.sleep(0.25)
        if isinstance(i, Path):
            i.unlink()


if __name__ == "__main__":
    multiprocessing.get_context("forkserver")
    main()
