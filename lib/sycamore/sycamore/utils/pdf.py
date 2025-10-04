import os
import select
import logging

from io import BytesIO
from PIL import Image
from pathlib import Path
from queue import Queue
from subprocess import PIPE, Popen, TimeoutExpired
from threading import Thread
from typing import Generator, IO, Iterator
from sycamore.utils.time_trace import LogTime


HEADER_BYTES = 40
TIMEOUT_SEC = 60
TIMEOUT_MSEC = 1000 * TIMEOUT_SEC

g_logger = logging.getLogger(__name__)


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
                g_logger.debug(f"reading. have {len(data)}/{need_bytes}")
                part = fh.read(need_bytes - len(data))
                if part == b"":  # Eof
                    assert len(data) == 0  # nothing left
                    break
                data = data + part
            else:
                g_logger.debug(f"no reading. have {len(data)}/{need_bytes}")

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
                g_logger.warning(f"pdftoppm stderr: {e}")
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


def convert_from_path_streamed_batched(filename: str, batch_size: int) -> Generator[list[Image.Image], None, None]:
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
                g_logger.debug(f"reading. have {len(data)}/{need_bytes}")
                part = fh.read(need_bytes - len(data))
                if part == b"":  # Eof
                    assert len(data) == 0  # nothing left
                    break
                data = data + part
            else:
                g_logger.debug(f"no reading. have {len(data)}/{need_bytes}")

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
                file.flush()
            q.put(out_path)
            logging.info(f"Write file into {out_path}")

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
                g_logger.warning(f"pdftoppm stderr: {e}")
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


class NonBlockReader:
    """
    This provides services to PdfToImageFiles below:
    1. Makes sure reads do not block or cause deadlocks across streams
    2. Makes sure the file is in the poller until it reaches EOF
    3. Buffers data as it's read, allowing callers to peek at it
    4. Resets the buffer when a range of bytes are done being processed
    """

    def __init__(self, stream: IO[bytes], poller: select.poll) -> None:
        self.stream = stream
        self.poller = poller
        os.set_blocking(stream.fileno(), False)
        poller.register(stream.fileno(), select.POLLIN)
        self.buf = BytesIO()
        self.view = memoryview(bytearray(64 * 1024))
        self.eof = False

    def do_read(self) -> bool:
        "Returns if the caller is advised to poll for additional input."
        if self.eof:
            return True  # allows us to poll on the other stream
        nbytes = self.stream.readinto(self.view)  # type: ignore[attr-defined]
        if nbytes:
            self.buf.write(self.view[0:nbytes])
            return False
        elif nbytes == 0:
            self.poller.unregister(self.stream.fileno())
            self.eof = True
            return False
        return True

    def reset(self, nbytes: int) -> None:
        old = self.buf.getbuffer()
        self.buf = BytesIO(old[nbytes:])  # move trailing bytes to front
        self.buf.seek(0, os.SEEK_END)

    def log(self, prefix: str) -> None:
        for bline in self.buf.readlines():
            line = bline.rstrip().decode()
            g_logger.warning(f"{prefix}{line}")


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
        self.image_num = 0
        self.timer = LogTime("convert_to_image")
        self.timer.start()
        args = ["pdftoppm", "-r", str(resolution), self.pdf_path]
        self.proc = Popen(args, bufsize=0, stdout=PIPE, stderr=PIPE)

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
                g_logger.warning(f"pdftoppm {proc.pid} did not terminate; killing")
                proc.kill()
                proc.wait(1)
            self.timer.measure()

    def __iter__(self) -> Iterator[Path]:
        assert self.in_context, "Use PdfToImageFiles as a context manager"
        proc = self.proc
        assert proc.stdout and proc.stderr
        poller = select.poll()
        out = NonBlockReader(proc.stdout, poller)
        err = NonBlockReader(proc.stderr, poller)
        image_size = -1  # indicates we haven't parsed header yet

        while not (out.eof and err.eof):
            out_should_poll = out.do_read()
            err_should_poll = err.do_read()
            if out_should_poll and err_should_poll:
                if not poller.poll(TIMEOUT_MSEC):  # avoid hangs
                    err.log("pdftoppm stderr: ")
                    raise TimeoutError(f"pdftoppm {self.pdf_path}")
                continue

            if image_size < 0:
                image_size = self.try_parse_header(out)

            if image_size >= 0:
                if out_path := self.try_write_image(out, image_size):
                    image_size = -1
                    yield out_path

        err.log("pdftoppm stderr: ")
        with LogTime("wait_for_pdftoppm_to_exit", log_start=True):
            proc.wait(TIMEOUT_SEC)  # just in case; should be instant
        self.timer.measure()

        assert proc.returncode is not None
        if proc.returncode != 0:
            msg = err.buf.getvalue().decode()
            raise ValueError(f"pdftoppm failed {proc.returncode}.  All stderr:{msg}")

    def try_parse_header(self, out: NonBlockReader) -> int:
        "Returns image size in bytes if header parsed, else -1."
        if out.buf.tell() < HEADER_BYTES:
            return -1
        hdr = out.buf.getvalue()[0:HEADER_BYTES]
        code, size, rgb = hdr.split(b"\n")[0:3]
        width, height = [int(s) for s in size.split(b" ")]
        return len(code) + len(size) + len(rgb) + 3 + (width * height * 3)

    def try_write_image(self, out: NonBlockReader, image_size: int) -> Path | None:
        "Returns path if complete image was written out, else None."
        if out.buf.tell() < image_size:
            return None
        out_path = self.file_dir / f"image.{self.image_num}.ppm"
        with (
            open(out_path, "wb") as fp,
            out.buf.getbuffer() as view,
        ):
            fp.write(view[0:image_size])
        out.reset(image_size)  # slide to front
        self.image_num += 1
        return out_path
