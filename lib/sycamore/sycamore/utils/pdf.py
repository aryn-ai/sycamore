import logging
import os
import tempfile
from io import BytesIO
from queue import Queue
from subprocess import PIPE, Popen
from threading import Thread
from typing import List, Generator

from PIL import Image

from sycamore.utils.cache_manager import CacheManager
from sycamore.utils.time_trace import LogTime

pdf_to_ppm_cache = CacheManager(os.path.join(tempfile.gettempdir(), "SycamoreCache/PDFToPPMCache"))


def convert_from_path_streamed(pdf_path: str) -> Generator[Image.Image, None, None]:
    class StdoutEOF:
        pass

    class StderrEOF:
        pass

    def capture_exception(q, fn, finish_msg):
        try:
            fn()
        except Exception as e:
            q.put(e)

        q.put(finish_msg)

    def read_stdout(fh, q):
        HEADER_BYTES = 40
        need_bytes = HEADER_BYTES
        data = b""
        while True:
            if need_bytes > len(data):
                logging.debug(f"reading. have {len(data)}/{need_bytes}")
                part = fh.read(need_bytes - len(data))
                if part == b"":  # Eof
                    assert len(data) == 0  # nothing left
                    break
                data = data + part
            else:
                logging.debug(f"no reading. have {len(data)}/{need_bytes}")

            if len(data) < need_bytes:
                continue

            code, size, rgb = tuple(data[0:HEADER_BYTES].split(b"\n")[0:3])
            size_x, size_y = tuple(size.split(b" "))
            need_bytes = len(code) + len(size) + len(rgb) + 3 + int(size_x) * int(size_y) * 3

            if len(data) < need_bytes:
                continue

            img = Image.open(BytesIO(data[0:need_bytes])).convert("RGB")
            q.put(img)
            data = data[need_bytes:]
            need_bytes = HEADER_BYTES

    def read_stderr(fh, q):
        while True:
            line = fh.readline()
            if line == b"":
                break

            q.put(line.decode().rstrip())

    with LogTime("convert_to_image"):
        # If we don't do this, then if the stderr buffer fills up we could get stuck.
        # Popen.communicate() reads the entire strings.
        args = ["pdftoppm", "-r", "200", pdf_path]
        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        q: Queue = Queue(4)
        t_out = Thread(
            target=capture_exception, daemon=True, args=(q, lambda: read_stdout(proc.stdout, q), StdoutEOF())
        )
        t_out.start()
        t_err = Thread(
            target=capture_exception, daemon=True, args=(q, lambda: read_stderr(proc.stderr, q), StderrEOF())
        )
        t_err.start()

        more_out = True
        more_err = True
        stderr = []
        while more_out or more_err:
            e = q.get()
            if isinstance(e, Exception):
                raise e
            elif isinstance(e, Image.Image):
                yield e
            elif isinstance(e, str):
                logging.warning(f"pdftoppm stderr: {e}")
                stderr.append(e)
            elif isinstance(e, StdoutEOF):
                more_out = False
            elif isinstance(e, StderrEOF):
                more_err = False
            else:
                raise ValueError(f"Unexpected thing on queue: {e}")

        with LogTime("wait_for_pdftoppm_to_exit", log_start=True):
            proc.wait()

        assert proc.returncode is not None
        if proc.returncode != 0:
            raise ValueError(f"pdftoppm failed {proc.returncode}.  All stderr:{stderr}")

        t_out.join()
        t_err.join()


def convert_from_path_streamed_batched(
    filename: str, batch_size: int, file_checksum: str, use_cache: bool = True
) -> Generator[List[Image.Image], None, None]:
    """Note: model service will call this to get batches of images for processing"""
    images = pdf_to_ppm_cache.get(file_checksum) if use_cache else None
    if images:
        logging.info("PDFToPPM Cache Hit. Getting the results from cache.")
        yield from yield_cached_batches(images, batch_size)
    else:
        logging.info("PDFToPPM Cache Miss. Getting the results.")
        yield from yield_batches(filename, batch_size, use_cache, file_checksum)


def yield_cached_batches(images: list, batch_size: int) -> Generator[List[Image.Image], None, None]:
    batch = []
    for image in images:
        batch.append(image)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def yield_batches(
    filename: str, batch_size: int, use_cache: bool, hash_key: str
) -> Generator[List[Image.Image], None, None]:
    batch = []
    all_images = []
    for i in convert_from_path_streamed(filename):
        batch.append(i)
        if len(batch) == batch_size:
            yield batch
            all_images.extend(batch)
            batch = []

    if len(batch) > 0:
        all_images.extend(batch)
        yield batch

    pdf_to_ppm_cache.set(hash_key, all_images)
