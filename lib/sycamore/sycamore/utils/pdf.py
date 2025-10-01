import logging

from io import BytesIO
from PIL import Image
from pathlib import Path
from queue import Queue
from subprocess import PIPE, Popen, TimeoutExpired
from threading import Thread
from typing import List, Generator, Iterator
from sycamore.utils.time_trace import LogTime


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
        HEADER_BYTES = 40
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
        self.file_dir = Path(file_dir)
        self.in_cm = False
        self.timer = LogTime("convert_to_image")
        self.timer.start()
        args = ["pdftoppm", "-r", str(resolution), str(pdf_path)]
        self.proc = Popen(args, stdout=PIPE, stderr=PIPE)
        self.running = True
        self.q: Queue = Queue(1)
        self.t_out = Thread(target=self.read_stdout, name=f"ThrReadStdout-{pdf_path}", daemon=True)
        self.t_err = Thread(target=self.read_stderr, name=f"ThrReadStderr-{pdf_path}", daemon=True)
        self.t_out.start()
        self.t_err.start()

    def __enter__(self) -> "PdfToImageFiles":
        self.in_cm = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        proc = self.proc
        assert proc.stdout
        assert proc.stderr
        if proc.poll() is None:
            self.running = False
            proc.terminate()
            try:
                proc.wait(2)
            except TimeoutExpired:
                proc.kill()
                proc.wait(1)
            proc.stdout.close()  # will wake up readers
            proc.stderr.close()
            while not self.q.empty():
                self.q.get_nowait()  # avoid deadlock
            self.t_out.join()
            self.t_err.join()
            self.timer.measure()

    def __iter__(self) -> Iterator[Path]:
        assert self.in_cm, "PdfToImageFiles must be used as context manager"
        proc = self.proc
        more_out = True
        more_err = True
        stderr = []
        while more_out or more_err:
            e = self.q.get()
            if isinstance(e, Path):
                yield e
            elif isinstance(e, str):
                logging.warning(f"pdftoppm stderr: {e}")
                stderr.append(e)
            elif e == StdoutEof:
                more_out = False
            elif e == StderrEof:
                more_err = False
            elif isinstance(e, Exception):
                raise e
            else:
                raise ValueError(f"Unexpected thing on queue: {e}")

        with LogTime("wait_for_pdftoppm_to_exit", log_start=True):
            proc.wait()
        self.t_out.join()
        self.t_err.join()
        self.timer.measure()

        assert proc.returncode is not None
        if proc.returncode != 0:
            raise ValueError(f"pdftoppm failed {proc.returncode}.  All stderr:{stderr}")

    def read_stdout(self) -> None:
        fh = self.proc.stdout
        assert fh
        file_dir = self.file_dir
        HEADER_BYTES = 40
        need_bytes = HEADER_BYTES
        data = b""
        image_num = 0
        try:
            while True:
                got = len(data)
                if need_bytes > got:
                    want = need_bytes - len(data)
                    logging.debug(f"reading {want}; have {got}/{need_bytes}")
                    part = fh.read(want)
                    if not self.running:
                        return
                    if part == b"":  # Eof
                        assert got == 0  # nothing left
                        break
                    data += part  # FIXME: quadratic
                else:
                    logging.debug(f"no reading. have {got}/{need_bytes}")

                if len(data) < need_bytes:
                    continue

                code, size, rgb = data[0:HEADER_BYTES].split(b"\n")[0:3]
                size_x, size_y = [int(s) for s in size.split(b" ")]
                need_bytes = len(code) + len(size) + len(rgb) + 3 + (size_x * size_y * 3)

                if len(data) < need_bytes:
                    continue

                out_path = file_dir / f"image.{image_num}.ppm"
                with open(out_path, "wb") as file:
                    file.write(data[0:need_bytes])
                self.q.put(out_path)

                image_num += 1
                data = data[need_bytes:]
                need_bytes = HEADER_BYTES

        except Exception as e:
            self.q.put(e)
        self.q.put(StdoutEof)

    def read_stderr(self) -> None:
        fh = self.proc.stderr
        assert fh
        try:
            while True:
                line = fh.readline()
                if not self.running:
                    return
                if line == b"":
                    break
                self.q.put(line.decode().rstrip())
        except Exception as e:
            self.q.put(e)
        self.q.put(StderrEof)
