from os import PathLike
import sys
from typing import Optional, Callable, Any
import subprocess
from pathlib import Path

from sycamore.data.document import Document
from sycamore.docset import DocSet

__this = sys.modules[__name__]


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


def view_pdf(
    docset: DocSet,
    doc_to_url: Optional[Callable[[Document], str]] = None,
    doc_to_display_name: Optional[Callable[[Document], str]] = None,
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

    if doc_to_url is None:
        assert hasattr(
            __this, "_doc_to_url"
        ), "doc_to_url must either be provided as an arg to view_pdf or init_viewpdf"
        doc_to_url = __this._doc_to_url
    if doc_to_display_name is None:
        assert hasattr(
            __this, "_doc_to_display_name"
        ), "doc_to_display_name must either be provided as an arg to view_pdf or init_viewpdf"
        doc_to_display_name = __this._doc_to_display_name
    assert doc_to_url is not None, "Type narrowing. This ought to be unreachable"
    assert doc_to_display_name is not None, "Type narrowing. This ought to be unreachable"

    # Add url and display name props, and get the minimal data to show
    docset = docset.with_property("_vpdf_url", doc_to_url).with_property("_vpdf_name", doc_to_display_name)
    props.extend(["_vpdf_url", "_vpdf_name"])
    info = []
    for doc in docset.take_stream():
        info.append({p: dotted_lookup(doc.properties, p) for p in props})
    info.sort(key=lambda i: i["_vpdf_name"])

    html_frags = _view_pdf_html(info, max_inline, height, props)
    # Multiple iframes didn't seem to work so we do them one at a time
    for frag in html_frags:
        display(HTML(frag))


def _view_pdf_html(info: list[dict[str, Any]], max_inline: int, height: int, props: list[str]) -> list[str]:
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


def init_viewpdf(
    doc_to_url: Optional[Callable[[Document], str]] = None,
    doc_to_display_name: Optional[Callable[[Document], str]] = None,
):
    """
    Initializes global default doc_to_url and doc_to_display_name functions for
    later use in `view_pdf`

    Args:
        doc_to_url: A function that takes a Document and returns a URL.
        doc_to_display_name: A function that takes a Document and returns a display name.
    """
    # Explanation: __this is a reference to this module. This function effectively adds
    # the provided functions to the module - view_pdf can then access them as defaults.
    global __this

    assert doc_to_url is not None or doc_to_display_name is not None
    if doc_to_url is not None:
        __this._doc_to_url = doc_to_url  # type: ignore
    if doc_to_display_name is not None:
        __this._doc_to_display_name = doc_to_display_name  # type: ignore


def local_files(
    root_dir: PathLike = Path.home(),
    port: int = 8086,
) -> Optional[subprocess.Popen]:
    """
    Initializes view_pdf for use with local files. Since jupyter doesn't like to open
    `file://` urls, we run a file server in a subprocess to serve them via http. This
    function calls `init_viewpdf`, so future calls to `view_pdf` work as expected.

    Args:
        root_dir: The root directory to serve files from. This must be a parent directory
            of every file that might want to be viewed. Default is the user's home directory.
        port: The port to serve files on. Default is 8086

    Returns:
        The subprocess running the file server. If the server is already running (you've
        already called local_files() in this process), returns nothing.
    """
    import socket

    def doc_to_url(doc: Document) -> str:
        path = doc.properties["path"]
        rd_str = str(root_dir)
        assert str(path).startswith(
            rd_str
        ), f"Document with path {path} does not live within {rd_str}, so cannot serve it"
        return f"http://localhost:{port}/{path[len(rd_str) + 1:]}"

    def doc_to_name(doc: Document) -> str:
        return Path(doc.properties["path"]).name

    init_viewpdf(doc_to_url, doc_to_name)
    server_is_running = True
    s = None
    try:
        s = socket.create_connection(("localhost", port), timeout=0.01)
    except Exception:
        server_is_running = False
    finally:
        if s is not None:
            s.close()

    if not server_is_running:
        server = subprocess.Popen(["python", "-m", "http.server", str(port)], cwd=root_dir)
        return server
    return None
