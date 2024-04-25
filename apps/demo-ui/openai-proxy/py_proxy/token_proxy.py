from gevent import monkey

# ruff: noqa: E402
monkey.patch_all()  # Must be before other imports

import os
import sys
import logging
import requests
import warnings
import urllib3

from gevent.socket import socket
from gevent.pywsgi import WSGIServer
from flask import abort, current_app, Flask, make_response, redirect, request, url_for
from flask_cors import CORS

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

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


@app.route("/tok/<tok>")
def proxy_token(tok="", arg=""):
    token = current_app.config["token"]
    if tok != token:
        logging.warning(f"bad token {tok}")
        abort(403)

    resp = redirect(url_for("proxy_plain"))
    resp.set_cookie("token", token)
    return resp


@app.route("/", methods=METHODS)
@app.route("/<path:arg>", methods=METHODS)
def proxy_plain(arg=""):
    tok = request.cookies.get("token")
    if not tok:
        logging.warning("missing token")
        abort(403)
    token = current_app.config["token"]
    if tok != token:
        logging.warning(f"wrong token {tok}")
        abort(403)

    base = current_app.config["base"]
    url = base + arg
    reply = requests.request(
        method=request.method,
        params=request.args,
        url=url,
        headers=request.headers,
        verify=False,
    )

    headers = [(k, v) for k, v in reply.headers.items() if k.lower() not in BAD_HEADERS]

    resp = make_response(reply.content, reply.status_code, headers)
    return resp


@app.route("/healthz")
def healthz():
    return "OK"


def main(args=None):
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

    sock = socket()
    sock.bind(("0.0.0.0", port))
    sock.listen()

    print(f"Bound token server to port {port}")
    if uid == 0:
        os.setuid(UID)
        print(f"Changed uid to {UID}")

    # Use gevent WSGIServer for asynchronous behavior
    server = WSGIServer(sock, app, certfile=f"{host}-cert.pem", keyfile=f"{host}-key.pem")
    if port == 443:
        print(f"Serve https://{host}/tok/{token}")
    else:
        print(f"Serve https://{host}:{port}/tok/{token}")
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
