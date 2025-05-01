from os import PathLike
from typing import Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod

from sycamore.data.document import Document
from sycamore.docset import DocSet


def slow_pprint(
    v: object, max_bytes: Optional[int] = None, width: int = 120, chunk_size: int = 1000, delay: float = 0.25
) -> None:
    """
    Prints large outputs slowly to prevent Jupyter from dropping output.

    Args:
        v: Value to be pretty-printed.
        max_bytes: Maximum number of bytes to display, None for no limit.
        width: Width for pretty formatting.
        chunk_size: Number of characters to print at a time.
        delay: Time (in seconds) to wait between chunks.
    """
    from devtools import PrettyFormat
    import time

    s = PrettyFormat(width=width)(v)
    if max_bytes is not None:
        s = s[:max_bytes]

    for i in range(0, len(s), chunk_size):
        print(s[i : i + chunk_size], end="", flush=True)
        time.sleep(delay)
    if not s.endswith("\n"):
        print("", flush=True)


def bound_memory(gb: int = 4) -> None:
    """
    Limits the process's memory usage.

    Args:
        gb: Memory limit in gigabytes.
    """
    import resource
    import platform

    if platform.system() != "Linux":
        print("WARNING: Memory limiting only works on Linux.")

    limit_bytes: int = gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, resource.RLIM_INFINITY))


def reimport(module_name: str) -> None:
    """
    Dynamically reloads a module.

    Args:
        module_name: Name of the module as a string.
    """
    import importlib

    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        print(f"Warning: You need to re-execute any statements like: `from {module_name} import ...`")

    except ModuleNotFoundError:
        print(f"Error: Module '{module_name}' not found.")

    except Exception as e:
        print(f"Error reloading module '{module_name}': {e}")


class PDFViewer(ABC):
    """
    Base class for viewing PDF files in a jupyter notebook.
    """

    @abstractmethod
    def doc_to_url(self, doc: Document) -> str:
        pass

    @abstractmethod
    def doc_to_display_name(self, doc: Document) -> str:
        pass

    def __call__(
        self,
        docset: DocSet,
        max_inline: int = 1,
        height: int = 1000,
        props: list[str] = [],
    ) -> None:
        """Passthrough to ``view_pdf``"""
        self.view_pdf(docset, max_inline, height, props)

    def view_pdf(
        self,
        docset: DocSet,
        max_inline: int = 1,
        height: int = 1000,
        props: list[str] = [],
    ) -> None:
        """
        View the pdfs in a docset using jupyter's display functionality. If there are
        more documents in the docset than max_inline, shows a list with links to open
        the pdfs in browser tabs. Otherwise, display them inline in iframes.

        Args:
            docset: the docset whose pdfs to show
            doc_to_url: Function to generate a url from which to download the pdf. Must be
                set either here or in `init_viewpdf`.
            doc_to_display_name: Function to generate a display name for each document.
                Must be set either here or in `init_viewpdf`
            max_inline: The maximum number of documents to show inline in iframes. Default
                is 1. If there are more documents than max_inline, view_pdf displays a list
                of links to open the pdfs in new browser tabs.
            height: The height of the iframe elements that will be rendered
            props: A list of properties to show for each document. Default is no properties.
        """
        from sycamore.utils.nested import dotted_lookup
        from IPython.display import HTML, display

        # Add url and display name props, and get the minimal data to show
        docset = docset.with_property("_vpdf_url", self.doc_to_url).with_property(
            "_vpdf_name", self.doc_to_display_name
        )
        props.extend(["_vpdf_url", "_vpdf_name"])
        info = []
        for doc in docset.take_stream():
            info.append({p: dotted_lookup(doc.properties, p) for p in props})
        info.sort(key=lambda i: i["_vpdf_name"])

        html_frags = self._view_pdf_html(info, max_inline, height, props)
        # Multiple iframes didn't seem to work so we do them one at a time
        for frag in html_frags:
            display(HTML(frag))

    def _view_pdf_html(self, info: list[dict[str, Any]], max_inline: int, height: int, props: list[str]) -> list[str]:
        # Build html to display - if more than max_inline docs,
        # make a list with links, otherwise, inline them as iframes
        def html_prop_list(inf: dict) -> list[str]:
            if len(inf) == 0:
                return []
            ans = []
            ans.append("<ul>")
            for k, v in inf.items():
                ans.append(f"<li>{k}: {v}</li>")
            ans.append("</ul>")
            return ans

        html_frags = []
        if len(info) <= max_inline:
            html_frags.append(f"<h1>{len(info)} documents. Displaying inline</h1>")
            f = []
            for inf in info:
                f.append(f"<h2>{inf.pop('_vpdf_name')}</h2>")
                url = inf.pop("_vpdf_url")
                f.extend(html_prop_list(inf))
                f.append(f'<iframe src="{url}" width="90%" height="{height}" />')
                html_frags.append("\n".join(f))
                f = []
        else:
            html_frags.append(f"<h1>{len(info)} documents. Links will open a new tab</h1><ul>")
            for inf in info:
                name = inf.pop("_vpdf_name")
                url = inf.pop("_vpdf_url")
                html_frags.append(f'<li> <a href="{url}" target="_blank">{name}</a>')
                html_frags.extend(html_prop_list(inf))
            html_frags.append("</ul>")
            html_frags = ["\n".join(html_frags)]

        return html_frags


class LocalFileViewer(PDFViewer):
    """
    Implementation of PDFViewer for local files. Jupyter sandboxes the filesystem,
    so `file:///path/to/file` links do not work - accordingly, we start a simple
    http server and generate http urls to hit it.

    Args:
        root_dir: The root directory to serve files from. Default is the user's home
            directory. Must be a parent of any files you want to view.
        port: port to start the file server on. If unspecified, we will pick one at random.
            If specified and the port is in use, we assume that the thing at that port
            will serve files.
        suppress_used_port_warning: Do not emit a warning when providing a port that is already
            in use. This is used when ser/de-ing this object due to ray. Users should never care.
    """

    def __init__(
        self, root_dir: PathLike = Path.home(), port: Optional[int] = None, *, suppress_used_port_warning: bool = False
    ):
        import logging
        import subprocess

        self._root = root_dir

        self._subprocess = None
        should_start_server = self._set_port(port, suppress_used_port_warning)
        if should_start_server:
            logging.warning(f"Running file server on localhost port {self._port}")
            self._subprocess = subprocess.Popen(
                ["python", "-m", "http.server", "-b", "127.0.0.1", str(self._port)], cwd=self._root
            )

    def _set_port(self, port: Optional[int], suppress_used_port_warning: bool) -> bool:
        import socket
        import random
        import logging

        if port is not None:
            self._port = port
            try:
                s = socket.create_server(("localhost", port))
                s.close()
                return True
            except Exception:
                if not suppress_used_port_warning:
                    logging.warning(
                        f"Port {port} is already in use. Will assume that it is an already-running LocalFileViewer server."
                    )
                return False
        else:
            max_tries = 10
            to_check = random.sample(range(1025, 65536), max_tries)
            for port in to_check:
                try:
                    socket.create_server(("localhost", port)).close()
                    self._port = port
                    return True
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to find an open port after {max_tries} tries. Checked: {to_check}. Go enter the lottery."
            )

    def __del__(self):
        # On GC, kill my subprocess.
        if self._subprocess is not None:
            self._subprocess.kill()

    def __reduce__(self):
        def deser(kwargs):
            return LocalFileViewer(**kwargs)

        kwargs = {"root_dir": self._root, "port": self._port, "suppress_used_port_warning": True}
        return deser, (kwargs,)

    def doc_to_url(self, doc: Document) -> str:
        path = doc.properties["path"]
        rd_str = str(self._root)
        assert str(path).startswith(
            rd_str
        ), f"Document with path {path} does not live within {rd_str}, so cannot serve it"
        return f"http://localhost:{self._port}/{path[len(rd_str) + 1:]}"

    def doc_to_display_name(self, doc: Document) -> str:
        return Path(doc.properties["path"]).name
