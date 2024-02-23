import time

from flask import Flask, request, jsonify, Response, send_file
import gevent

from gevent import monkey
from urllib3.exceptions import NewConnectionError

#monkey.patch_all()

from gevent.pywsgi import WSGIServer
import urllib3
import requests
import os
import sys
from flask_cors import CORS
from werkzeug.datastructures import Headers
import io
import logging
import boto3 

requests.packages.urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask('proxy', static_folder=None)
HOST='localhost'
PORT=3000
try:
    HOST=sys.argv[1]
    PORT=int(sys.argv[2])
except:
    pass

CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from http://localhost:3001 to any route

# Replace this with your actual OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# API endpoint of the OpenAI service
OPENAI_API_BASE = "https://api.openai.com"

OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_URL = f"https://{OPENSEARCH_HOST}:9200/"

UI_HOST = os.environ.get("LOAD_BALANCER", "localhost")
UI_BASE = f"https://{UI_HOST}:3001"

# AWS defaults
AWS_REGION = "us-east-1"

badHeaders = [
    'content-encoding',
    'content-length',
    'transfer-encoding',
    'connection',
]

# qa_logger = logging.getLogger("qa_log")
# qa_logger.setLevel(logging.WARNING)
# qalfh = logging.FileHandler("qa.log")
# qalfh.setLevel(logging.INFO)
# qalfh.setFormatter(logging.Formatter('[%(asctime)s]\t[%(levelname)s]\t%(message)s'))
# qa_logger.addHandler(qalfh)


def optionsResp(methods: str):
    # Respond to CORS preflight request
    resp = jsonify({})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = methods
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return resp


@app.route('/v1/completions', methods=['POST', 'OPTIONS'])
@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def proxy_stream_request():
    if request.method == 'OPTIONS':
        return optionsResp('POST')

    headers = Headers()
    headers["Content-Type"] = "application/json"
    headers['Authorization'] = f"Bearer {OPENAI_API_KEY}"

    # Forward the incoming request to OpenAI API with stream enabled
    response = requests.post(
        url=f"{OPENAI_API_BASE}{request.path}",
        headers=headers,
        json=request.json,
        stream=True,  # Enable streaming
        verify=False,
    )
    
    print(f"Outgoing Request - URL: {response.url}, Status Code: {response.status_code}")

    # Check if the response is a streaming response
    is_streaming_response = 'Transfer-Encoding' in response.headers and response.headers['Transfer-Encoding'] == 'chunked'

    if is_streaming_response:
        print("Streaming response detected")
        # Stream the OpenAI API response back to the client
        def stream_response():
            def generate_chunks():
                for chunk in response.iter_content(chunk_size=1024):
                    yield chunk

            return Response(generate_chunks(), response.status_code, response.headers.items())

        return stream_response()
    else:
        # Return the non-streaming response as a complete JSON object
        return (response.content, response.status_code, response.headers.items())

@app.route('/v1/pdf', methods=['POST', 'OPTIONS'])
def proxy():
    path = "pdf"
    if request.method == 'OPTIONS':
        return optionsResp('POST')

    url = request.json.get('url')
    if url.startswith('/'):
        source = url
    elif url.startswith("s3://"):
        trimmed_uri = url[5:]
        bucket_name, file_key = trimmed_uri.split('/', 1)
        s3 = boto3.client('s3', AWS_REGION)
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        source = response['Body']
    else:
        response = requests.get(url=url, verify=False)
        source = io.BytesIO(response.content)

    download_name = os.path.basename(url)
    
    return send_file(
        source,
        mimetype='application/pdf',
        as_attachment=True,
        download_name = download_name
    )

@app.route('/opensearch/<path:os_path>', methods=['GET','POST','PUT','DELETE','HEAD','OPTIONS'])
def proxy_opensearch(os_path):
    if request.method == 'OPTIONS':
        return optionsResp('GET, POST, PUT, DELETE, HEAD')
    
    # log = request.method + " " + OPENSEARCH_URL + os_path
    # if request.is_json and request.content_length is not None:
    #     print(request.json)
    #     log += " " + str(request.json)

    response = requests.request(
        method=request.method,
        params=request.args,
        url=OPENSEARCH_URL + os_path,
        json=request.json if (request.is_json and not request.content_length is None) else None,
        headers=request.headers,
        verify=False,
    )
    # qa_logger.info(log)
    # qa_logger.info(str(response.json()))

    return response.json()


@app.route('/opensearch-version', methods=['GET', 'OPTIONS'])
def opensearch_version(retries=3):
    if request.method == 'OPTIONS':
        return optionsResp('GET')
    try:
        response = requests.request(
            method='GET',
            url=OPENSEARCH_URL
        )
        return response.json()['version']['number'], 200
    except Exception as e:
        if retries <= 0:
            print(f"OpenSearch not standing at {OPENSEARCH_URL}. Out of retries.")
            return "OpenSearch not found", 503
        print(f"OpenSearch not standing at {OPENSEARCH_URL}. Retrying in 1 sec. {retries-1} retries left.")
        time.sleep(1)
        return opensearch_version(retries=retries-1)


@app.route('/', methods=['GET', 'OPTIONS'])
@app.route('/manifest.json', methods=['GET', 'OPTIONS'])
@app.route('/static/<path:arg>', methods=['GET', 'OPTIONS'])
@app.route('/viewPdf', methods=['GET', 'OPTIONS'])
@app.route('/favicon.ico', methods=['GET', 'OPTIONS'])
@app.route('/<arg>.png', methods=['GET', 'OPTIONS'])
@app.route('/robots.txt', methods=['GET', 'OPTIONS'])
def proxy_ui(arg=None):
    if request.method == 'OPTIONS':
        return optionsResp('GET')

    resp = requests.request(
        method=request.method,
        params=request.args,
        url=UI_BASE + request.path,
        headers=request.headers,
        verify=False,
    )

    headers = [
        (k, v) for k, v in resp.headers.items()
        if k.lower() not in badHeaders
    ]

    return (resp.content, resp.status_code, headers)

@app.route('/healthz', methods=['GET'])
def healthz(arg=None):
    return 'OK'

if __name__ == '__main__':
    # Use gevent WSGIServer for asynchronous behavior
    if os.environ.get("SSL", "1") == "0":
        print("Proxy not serving over SSL.")
        http_server = WSGIServer(('0.0.0.0', PORT), app)
    else:
        http_server = WSGIServer(('0.0.0.0', PORT), app,
                                 certfile=f"{HOST}-cert.pem",
                                 keyfile=f"{HOST}-key.pem")
    print(f"Serving on {PORT}...")
    http_server.serve_forever()
