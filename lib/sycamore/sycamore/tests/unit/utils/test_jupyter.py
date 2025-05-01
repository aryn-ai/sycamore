from sycamore.utils.jupyter import LocalFileViewer
import socket
import time
import gc


def try_start_server(port: int, xfail: bool):
    if xfail:
        conflicted = False
        for _ in range(100):
            time.sleep(0.1)
            try:
                socket.create_server(("localhost", port)).close()
            except OSError:
                conflicted = True
                break
        assert conflicted
    else:
        # This branch is only really meaningful if called after
        # xfail=True, to test that stuff was cleaned up properly.
        time.sleep(0.1)
        socket.create_server(("localhost", port)).close()


def test_localfileviewer_no_leaks():
    port = 2647
    viewpdf = LocalFileViewer(port=port)
    try_start_server(port, xfail=True)

    newport = 2648
    viewpdf = LocalFileViewer(port=newport)
    # gc the original viewpdf
    gc.collect()
    try_start_server(port, xfail=False)
    try_start_server(newport, xfail=True)

    del viewpdf
    gc.collect()
    try_start_server(newport, xfail=False)
