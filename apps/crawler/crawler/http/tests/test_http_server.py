import http.server
import os
import socketserver

IPADDR = "127.0.0.1"
PORT = 13756

os.chdir("http_serve")
Handler = http.server.SimpleHTTPRequestHandler

Handler.extensions_map = {
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".qdf": "application/pdf",
}

httpd = socketserver.TCPServer((IPADDR, PORT), Handler)

print("serving at port", PORT)

httpd.serve_forever()
