import os
import select
import logging

from io import BytesIO
from PIL import Image
from pathlib import Path
from queue import Queue
from subprocess import PIPE, Popen, TimeoutExpired
from threading import Thread
from typing import List, Generator, Iterator
from sycamore.utils.time_trace import LogTime


HEADER_BYTES = 40


def convert_from_path_streamed(pdf_path: str) -> Generator[Image.Image, None, None]:
    """Deprecated. Switch to pdf_to_image_files"""

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
            target=capture_exception,
            name=f"ThrStdoutCaptureOld-{pdf_path}",
            args=(q, lambda: read_stdout(proc.stdout, q), StdoutEOF()),
            daemon=True,
        )
        t_out.start()
        t_err = Thread(
            target=capture_exception,
            name=f"ThrStderrCaptureOld-{pdf_path}",
            args=(q, lambda: read_stderr(proc.stderr, q), StderrEOF()),
            daemon=True,
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


def convert_from_path_streamed_batched(filename: str, batch_size: int) -> Generator[List[Image.Image], None, None]:
    """Deprecated. Switch to pdf_to_image_files
    Note: model service will call this to get batches of images for processing"""
    batch = []
    for i in convert_from_path_streamed(filename):
        batch.append(i)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


def pdf_to_image_files(pdf_path: str, file_dir: Path, resolution: int = 200) -> Generator[Path, None, None]:
    """Deprecated.  Switch to PdfToImageFiles"""

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
        need_bytes = HEADER_BYTES
        data = b""
        image_num = 0
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

            out_path = file_dir / f"image.{image_num}.ppm"
            with open(out_path, "wb") as file:
                file.write(data[0:need_bytes])
            q.put(out_path)

            image_num = image_num + 1

            data = data[need_bytes:]
            need_bytes = HEADER_BYTES

    def read_stderr(fh, q):
        while True:
            line = fh.readline()
            if line == b"":
                break

            q.put(line.decode().rstrip())

    assert isinstance(file_dir, Path)

    with LogTime("convert_to_image"):
        # If we don't have the separate threads for reading stdout/stderr,
        # then if the stderr buffer fills up we could get stuck.
        args = ["pdftoppm", "-r", str(resolution), pdf_path]
        proc = Popen(args, stdout=PIPE, stderr=PIPE)
        q: Queue = Queue(1)
        t_out = Thread(
            target=capture_exception,
            name=f"ThrStdoutCapture-{pdf_path}",
            args=(q, lambda: read_stdout(proc.stdout, q), StdoutEOF()),
            daemon=True,
        )
        t_out.start()
        t_err = Thread(
            target=capture_exception,
            name=f"ThrStderrCapture-{pdf_path}",
            args=(q, lambda: read_stderr(proc.stderr, q), StderrEOF()),
            daemon=True,
        )
        t_err.start()

        more_out = True
        more_err = True
        stderr = []
        while more_out or more_err:
            e = q.get()
            if isinstance(e, Exception):
                raise e
            elif isinstance(e, Path):
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


class StdoutEof:
    "sentinel class"


class StderrEof:
    "sentinel class"


class PdfToImageFiles:
    """
    Writes the files (streamed) into file_dir.  Caller is responsible for calling
    path.unlink() to cleanup the files.

    Note: model service will call this to get images for processing
    """

    def __init__(self, *, pdf_path: str | Path, file_dir: str | Path, resolution: int = 200) -> None:
        self.pdf_path = str(pdf_path)
        self.file_dir = Path(file_dir)
        self.in_context = False
        self.timer = LogTime("convert_to_image")
        self.timer.start()
        args = ["pdftoppm", "-r", str(resolution), self.pdf_path]
        self.proc = Popen(args, stdout=PIPE, stderr=PIPE)

    def __enter__(self) -> "PdfToImageFiles":
        self.in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.in_context = False  # avoid reuse
        proc = self.proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(2)
            except TimeoutExpired:
                logging.warning(f"pdftoppm {proc.pid} did not terminate; killing")
                proc.kill()
                proc.wait(1)
            self.timer.measure()

    def __iter__(self) -> Iterator[Path]:
        assert self.in_context, "Use PdfToImageFiles as a context manager"
        file_dir = self.file_dir
        proc = self.proc
        stdout = proc.stdout
        stderr = proc.stderr
        assert stdout
        assert stderr
        outfd = stdout.fileno()
        errfd = stderr.fileno()
        os.set_blocking(outfd, False)
        os.set_blocking(errfd, False)
        poller = select.poll()
        poller.register(outfd, select.POLLIN)
        poller.register(errfd, select.POLLIN)

        outbuf = BytesIO()
        errbuf = BytesIO()
        outview = memoryview(bytearray(131072))  # 128kB
        errview = memoryview(bytearray(8192))  # 8kB
        more_out = True
        more_err = True
        seen_header = False
        need_bytes = 0
        image_num = 0

        while more_out or more_err:
            # readinto returns 0 for EOF; None if no bytes available
            if more_out:
                out_bytes = stdout.readinto(outview)  # type: ignore[attr-defined]
            else:
                out_bytes = None
            if more_err:
                err_bytes = stderr.readinto(errview)  # type: ignore[attr-defined]
            else:
                err_bytes = None
            if (out_bytes is None) and (err_bytes is None):
                nfds = poller.poll(60 * 1000)  # one minute is too long to wait
                if nfds == 0:
                    raise TimeoutError(f"pdftoppm {self.pdf_path}")
                continue

            if err_bytes == 0:
                more_err = False
                poller.unregister(errfd)
            elif err_bytes:
                errbuf.write(errview[0:err_bytes])

            if out_bytes == 0:
                more_out = False
                poller.unregister(outfd)
            elif out_bytes:
                outbuf.write(outview[0:out_bytes])

            if not seen_header:
                if outbuf.tell() < HEADER_BYTES:
                    continue
                hdr = outbuf.getvalue()[0:HEADER_BYTES]
                code, size, rgb = hdr.split(b"\n")[0:3]
                width, height = [int(s) for s in size.split(b" ")]
                need_bytes = len(code) + len(size) + len(rgb) + 3 + (width * height * 3)
                seen_header = True
                # !!! fall through
            if seen_header:
                if outbuf.tell() < need_bytes:
                    continue
                out_path = file_dir / f"image.{image_num}.ppm"
                with (
                    open(out_path, "wb") as fp,
                    outbuf.getbuffer() as view,
                ):
                    fp.write(view[0:need_bytes])
                outbuf = BytesIO(outbuf.getbuffer()[need_bytes:])  # slide
                outbuf.seek(0, os.SEEK_END)
                seen_header = False
                image_num += 1
                yield out_path

        with LogTime("wait_for_pdftoppm_to_exit", log_start=True):
            proc.wait()
        self.timer.measure()

        for bmsg in errbuf.readlines():
            msg = bmsg.decode()
            logging.warning(f"pdftoppm stderr: {msg}")

        assert proc.returncode is not None
        if proc.returncode != 0:
            msg = errbuf.getvalue().decode()
            raise ValueError(f"pdftoppm failed {proc.returncode}.  All stderr:{msg}")
