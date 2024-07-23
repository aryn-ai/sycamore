import ssl
import random
import mimetypes
import http.client
import urllib.parse

from typing import Optional


class OneShotKaClient:
    """
    Acts like a keepalive client, but actually closes the connection at
    the end.  Basically, doesn't send Connection: close header as
    urllib.request does.  This makes it more like curl.
    """

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def __init__(self, url: str) -> None:
        self.url = url
        self.up = urllib.parse.urlparse(url)
        self.agent = "ArynKa/1.0"

    def set_agent(self, a: str) -> None:
        self.agent = a

    def get(self, *, headers: Optional[dict[str, str]] = None) -> bytes:
        hdr = {
            "User-Agent": self.agent,
            "Host": self.up.netloc,
            "Accept": "*/*",
            "Accept-Encoding": "identity",
        }
        if headers:
            hdr.update(headers)

        self._htconn()
        self.conn.request("GET", self.up.path, headers=hdr)
        return self._finish()

    def post(
        self,
        *,
        headers: Optional[dict[str, str]] = None,
        form: Optional[dict[str, str]] = None,
        cache: Optional[dict[str, bytes]] = None,
    ) -> bytes:
        bnd = self._boundary()
        hdr = {
            "User-Agent": self.agent,
            "Host": self.up.netloc,
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "Transfer-Encoding": "chunked",
            "Content-Type": f"multipart/form-data; boundary={bnd}",
        }
        if headers:
            hdr.update(headers)

        data = self.form2bytes(bnd, form, cache)
        self._htconn()
        self.conn.request("POST", self.up.path, body=data, headers=hdr, encode_chunked=True)
        return self._finish()

    def form2bytes(self, bnd: str, form: Optional[dict[str, str]], cache: Optional[dict[str, bytes]]) -> bytes:
        form = form if form else {}
        cache = cache if cache else {}
        ary = []
        for k, v in form.items():
            if v.startswith("@"):
                path = v[1:]
                buf = cache.get(path)
                if buf is None:
                    with open(path, "rb") as fp:
                        buf = fp.read()
                typ, enc = mimetypes.guess_type(path)
                fn = path.split("/")[-1]
                strs = [
                    f"--{bnd}",
                    f'Content-Disposition: form-data; name="{k}"; filename="{fn}"',
                    f"Content-Type: {typ}",
                ]
                if enc:
                    strs.append(f"Content-Type: {typ}")
                bary = [s.encode() for s in strs]
                bary.append(b"")
                bary.append(buf)
            else:
                bary = [
                    f"--{bnd}".encode(),
                    f'Content-Disposition: form-data; name="{k}"'.encode(),
                    b"",
                    v.encode(),
                ]
            ary.extend(bary)
        ary.append(f"--{bnd}--".encode())  # terminator
        ary.append(b"")
        return b"\r\n".join(ary)

    @property
    def resp(self) -> http.client.HTTPResponse:
        return self._resp

    @property
    def status(self) -> int:
        return self._resp.status

    @property
    def reason(self) -> str:
        return self._resp.reason

    def getheader(self, name, default=None) -> Optional[str]:
        return self._resp.getheader(name, default)

    def getheaders(self) -> list[tuple[str, str]]:
        return self._resp.getheaders()

    def _htconn(self) -> None:
        u = self.up
        assert u.hostname
        if u.scheme == "http":
            self.conn = http.client.HTTPConnection(u.hostname, u.port)
            return
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        self.conn = http.client.HTTPSConnection(u.hostname, u.port, context=ctx)

    def _finish(self) -> bytes:
        self._resp = self.conn.getresponse()
        buf = self._resp.read()  # all
        self.conn.close()
        del self.conn
        return buf

    def _boundary(self) -> str:
        return "".join(random.choice(self.alphabet) for i in range(22))
