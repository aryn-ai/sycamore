from gevent import monkey

# ruff: noqa: E402
monkey.patch_all()  # Must be before other imports

import os
import sys
import socket
import logging
import requests
import warnings
import urllib3

import gevent
from gevent.pywsgi import LoggingLogAdapter, WSGIServer
from flask import abort, current_app, Flask, make_response, redirect, request, url_for
from flask_cors import CORS

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("token")

# This needs to be global to make decorators work...
app = Flask("token_proxy", static_folder=None)

UID = 1000  # This is the Docker default
UI_PORT = 3000
METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]

BAD_HEADERS = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
}


@app.route("/healthz")
def healthz():
    return "OK"


@app.route("/tok/<tok>")
def proxy_token(tok="", arg=""):
    token = current_app.config["token"]
    if tok != token:
        logging.warning(f"bad token {tok}")
        abort(403)

    logger.info(f"setting token={tok}")
    resp = redirect(url_for("proxy_plain"))
    resp.set_cookie("token", token)
    return resp


@app.route("/", methods=METHODS)
@app.route("/<path:arg>", methods=METHODS)
def proxy_plain(arg=""):
    tok = request.cookies.get("token")
    if not tok:
        logger.warning("missing token")
        abort(403)
    token = current_app.config["token"]
    if tok != token:
        logger.warning(f"wrong token {tok}")
        abort(403)

    url = current_app.config["base"] + arg
    data = None if request.content_length is None else request.get_data()
    reply = requests.request(
        method=request.method,
        params=request.args,
        url=url,
        data=data,
        headers=request.headers,
        verify=False,
    )

    headers = [(k, v) for k, v in reply.headers.items() if k.lower() not in BAD_HEADERS]

    resp = make_response(reply.content, reply.status_code, headers)
    return resp


def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("TOKEN %(message)s"))
    wsgilog = logging.getLogger("wsgi")
    wsgilog.propagate = False
    wsgilog.addHandler(sh)
    adapter = LoggingLogAdapter(wsgilog)

    if args is None:
        args = sys.argv[1:]
    token = args.pop(0)
    host = args.pop(0)
    port = int(args.pop(0)) if args else 443

    if os.environ.get("SSL", "1") == "0":
        raise RuntimeError("token_proxy only supports SSL")

    uid = os.getuid()
    if (port < 1024) and (uid != 0):
        raise RuntimeError(f"must be root to bind to port {port}")

    global app
    CORS(app)  # Allow all routes for all domains
    app.config["token"] = token
    app.config["base"] = f"https://{host}:{UI_PORT}/"

    sock = gevent.socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen()

    logger.info(f"Bound token server to port {port}")
    if uid == 0:
        os.setuid(UID)
        logger.info(f"Changed uid to {UID}")

    # Use gevent WSGIServer for asynchronous behavior
    server = WSGIServer(
        sock, app, log=adapter, certfile=f"{host}-cert.pem", keyfile=f"{host}-key.pem"
    )
    if port == 443:
        logger.info(f"Serve https://{host}/tok/{token}")
    else:
        logger.info(f"Serve https://{host}:{port}/tok/{token}")
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
