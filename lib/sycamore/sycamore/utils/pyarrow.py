import logging
from pathlib import Path
from typing import Optional, Tuple

from pyarrow.fs import FileSystem

logger = logging.getLogger(__name__)


def infer_fs(path: str) -> Tuple[FileSystem, str]:
    import re

    if not re.match("^[a-z0-9]+://.", path):
        # pyarrow expects URIs, accepts /dir/path, but rejects ./dir/path
        # normalize everything to a URI.
        p = Path(path)
        if p.is_absolute():
            path = p.as_uri()
        else:
            path = p.absolute().as_uri()

    from pyarrow import fs

    (fs, root) = fs.FileSystem.from_uri(str(path))

    fs = maybe_use_anonymous_s3_fs(fs, root)

    return (fs, root)


def maybe_use_anonymous_s3_fs(fs: FileSystem, root: str) -> FileSystem:
    from pyarrow.fs import S3FileSystem

    if not isinstance(fs, S3FileSystem):
        return fs

    try:
        fs.get_file_info(root)
        return fs
    except OSError as e:
        logger.warning(f"Got error {e} trying to get file info on {root}, trying again in anonymous mode")

    new_fs = S3FileSystem(anonymous=True)
    try:
        new_fs.get_file_info(root)
        logger.info(f"Successfully read path {root} with anonymous S3")
        return new_fs
    except OSError as e:
        logger.warning(
            f"Got error {e} trying to get file info on {root} in anonymous mode. Using previous, non-working fs"
        )

    return fs


def cross_check_infer_fs(filesystem: Optional[FileSystem], path: str) -> Tuple[FileSystem, str]:
    if filesystem is None:
        return infer_fs(path)

    (f, p) = infer_fs(path)
    # ray allows you to specify a path like s3://bucket/object with the S3 Pyarrow filesystem
    # and will silently fix the path to be acceptable. Do something similar here.
    if isinstance(filesystem, f.__class__):
        path = str(p)
    else:
        logger.warning(
            f"path {path} infers a filesystem of class {f.__class__} and a path of {p}, but"
            + f" the specified filesystem is {filesystem.__class__}. Using the path unchanged."
        )

    return (filesystem, path)
