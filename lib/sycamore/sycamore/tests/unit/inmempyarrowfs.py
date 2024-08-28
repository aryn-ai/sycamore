import io

from pyarrow.fs import FileSystem, FileSelector, FileInfo, FileType


class InMemPyArrowFileSystem(FileSystem):
    def __init__(self):
        self._fs = {}

    def copy_file(self, src, dest):
        assert False, "unimplemented"

    def create_dir(self, path, *, recursive=True):
        assert isinstance(path, str)
        # We're blob-like, so create dir is noop.
        pass

    def delete_dir(self, path):
        assert False, "unimplemented"

    def delete_dir_contents(self, path, missing_dir_ok=False):
        assert isinstance(path, str)
        assert missing_dir_ok, "unimplemented"
        path = path + "/"
        todelete = []
        for k, v in self._fs.items():
            if k.startswith(path):
                todelete.append(k)

        for k in todelete:
            del self._fs[k]

    def delete_file(self, path):
        assert isinstance(path, str)
        assert path in self._fs
        del self._fs[path]

    def equals(self, other):
        assert False, "unimplemented"

    def get_file_info(self, p):
        if isinstance(p, str):
            if p not in self._fs:
                return FileInfo(str(p))

            # API docs claim we can leave mtime & size as None
            return FileInfo(str(p), type=FileType.File)

        assert isinstance(p, FileSelector)
        assert p.allow_not_found, "unimplemented"
        assert not p.recursive, "unimplemented"
        dir = p.base_dir + "/"
        dlen = len(dir)
        ret = []
        for k, v in self._fs.items():
            if not k.startswith(dir):
                continue
            if "/" in k[dlen:]:
                continue
            ret.append(FileInfo(str(k), type=FileType.File))

        return ret

    def move(self, src, dest):
        assert False, "unimplemented"

    def normalize_path(self, path):
        assert False, "unimplemented"

    def open_append_stream(self, path):
        assert False, "unimplemented"

    def open_input_file(self, path):
        assert False, "unimplemented"

    def open_input_stream(self, path):
        assert isinstance(path, str)
        assert path in self._fs
        f = self._fs[path]
        assert isinstance(f, bytes)
        return io.BytesIO(f)

    def open_output_stream(self, path):
        class OpenFile(io.BytesIO):
            def __init__(self, fs, name):
                self._fs = fs
                self._name = name
                super().__init__()

            def close(self):
                assert isinstance(self._fs[self._name], OpenFile)
                self._fs[self._name] = self.getvalue()
                super().close()

        assert isinstance(path, str)
        assert path not in self._fs, "overwrite unimplemented"
        self._fs[path] = OpenFile(self._fs, path)
        return self._fs[path]
