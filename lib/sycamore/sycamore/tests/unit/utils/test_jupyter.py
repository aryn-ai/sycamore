from sycamore.utils.jupyter import LocalFileViewer
import socket
import pytest
import time
import gc


def test_localfileviewer_no_leaks():
    port = 2647
    viewpdf = LocalFileViewer(port=port)

    # Wait for subprocess server to start
    time.sleep(0.1)
    with pytest.raises(Exception):
        socket.create_server(("localhost", port)).close()

    newport = 2648
    viewpdf = LocalFileViewer(port=newport)
    # gc the original viewpdf
    gc.collect()
    socket.create_server(("localhost", port)).close()

    time.sleep(0.1)
    with pytest.raises(Exception):
        socket.create_server(("localhost", newport)).close()

    del viewpdf
    gc.collect()
    socket.create_server(("localhost", newport)).close()
