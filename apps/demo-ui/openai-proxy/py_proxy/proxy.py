from gevent import monkey

# ruff: noqa: E402
monkey.patch_all()  # Must be before other imports

from gevent.pywsgi import LoggingLogAdapter, WSGIServer
import urllib3
import requests
import os
import sys
import time
import openai
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from werkzeug.datastructures import Headers
import io
import logging
import boto3
import warnings
import mimetypes
import anthropic 

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("proxy")

app = Flask("proxy", static_folder=None)
HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from http://localhost:3001 to any route

# Replace this with your actual OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# API endpoint of the OpenAI service
OPENAI_API_BASE = "https://api.openai.com"

# ANTHROPIC configs
anthropic_client = anthropic.Anthropic()

ANTHROPIC_RAG_PROMPT = ""
current_directory = os.path.dirname(__file__)
anthropic_rag_prompt_filepath = os.path.join(current_directory, "anthropic_rag_prompt.txt")
with open(anthropic_rag_prompt_filepath, "r") as file:
    ANTHROPIC_RAG_PROMPT = file.read()
print("Using anthropic prompt: " + ANTHROPIC_RAG_PROMPT)

OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_URL = f"https://{OPENSEARCH_HOST}:9200/"

UI_HOST = os.environ.get("LOAD_BALANCER", "localhost")
UI_BASE = f"https://{UI_HOST}:3001"

# AWS defaults
AWS_REGION = "us-east-1"

badHeaders = [
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
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
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = methods
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


@app.route("/v1/completions", methods=["POST", "OPTIONS"])
@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def proxy_stream_request():
    if request.method == "OPTIONS":
        return optionsResp("POST")

    headers = Headers()
    headers["Content-Type"] = "application/json"
    headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

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
    is_streaming_response = (
        "Transfer-Encoding" in response.headers and response.headers["Transfer-Encoding"] == "chunked"
    )

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


@app.route("/v1/pdf", methods=["POST", "OPTIONS"])
def proxy():
    if request.method == "OPTIONS":
        return optionsResp("POST")

    url = request.json.get("url")
    if url.startswith("/"):
        source = url
    elif url.startswith("s3://"):
        trimmed_uri = url[5:]
        bucket_name, file_key = trimmed_uri.split("/", 1)
        s3 = boto3.client("s3", AWS_REGION)
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        source = response["Body"]
    else:
        response = requests.get(url=url, verify=False)
        source = io.BytesIO(response.content)

    download_name = os.path.basename(url)

    return send_file(
        source,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=download_name,
    )


@app.route("/viewLocal/<path:arg>", methods=["GET", "OPTIONS"])
def proxy_local(arg):
    if request.method == "OPTIONS":
        return optionsResp("GET")
    path = "/" + arg
    if "/../" in path:  # Prevent filesystem snooping
        return "invalid path"
    type, enc = mimetypes.guess_type(path)
    if not type:
        type = "text/html"
    return send_file(path, mimetype=type)


def make_openai_call(messages, model_id="gpt-4"):
    response = openai.ChatCompletion.create(model=model_id, messages=messages, temperature=0.0, stream=False)

    return response


@app.route("/aryn/simplify", methods=["POST", "OPTIONS"])
def simplify_answer():
    if request.method == "OPTIONS":
        return optionsResp("POST")

    summarize_prompt = """
You are a post processor for a question answering system. Your job is to modify a system generated \
answer for a user question, based on the type of question it is.

Questions can be of the following types:
1. Part number lookup, e.g. What is the part number for the service kit for a Reba RL A5?
2. General information lookup, e.g. What pad compound should i use on my Centerline rotor?
3. Binary questions, e.g. Can I combine the clamp for my brake and my XX shifter?
4. Process lookups, e.g. How do I adjust the shifting on my XX shifter?
5. Other

The answer will either:
1. contain some numeric citations to search results, e.g. [1], [2], [5]
2. say it was unable to find the answer to your question

Your job is to do the following:
1. If the question is a process lookup and there is a valid answer, say "The answer can be found in search results \
[{citations}]", followed by a 1 sentence summary of the answer. Do not use more than 1 sentence.
2. If the answer  says it's unable to answer the question, or there is no information for the question, say "Unable \
to answer the question based on the search results. Please look at the top results to find relevant references"
3. Otherwise return the answer unmodified

If citations are surrounded by a dollar sign and curly braces, e.g. [${1}], change it to only be a number surrounded \
by square brackets, i.e. [1]

Examples:
Question: what is the part number I need for a service kit for my Reba RL A5?

System generated answer:
The answer can be found in search result(s) [1]. The part number for the service kit for a Reba RL A5 \
is 00.4315.032.650.

Answer:
The answer can be found in search result(s) [1]. The part number for the service kit for a Reba RL A5 \
is 00.4315.032.650.
----
Question: Are there different color lower leg options for my SID?


System generated answer:
The answer can be found in search result(s) [no citation given].
The search results do not provide information on different color options for the lower legs of a SID suspension fork. \
For accurate information, please refer to the product catalog or contact the manufacturer directly.

Answer:
Unable to answer the question based on the search results. Please look at the top results to find relevant references

----

Question: How do I adjust sag on my super deluxe coil?

System generated answer:
The answer can be found in search result(s) [4].
To adjust sag on your Super Deluxe Coil, follow these steps:
1. Install the coil spring and spring retainer.
2. Adjust the spring preload adjuster until the coil spring contacts the spring retainer. Ensure there is no vertical \
play between the coil spring and the retainer.
3. Do not exceed 5 mm (or five full turns of rotation) on the spring preload adjuster as this will damage the shock.
4. If more than 5 turns are necessary to achieve proper sag, use a higher weight spring.
5. If your target sag percentage is not achieved, spring preload adjustment and/or coil spring replacement must be \
performed.

Answer:
The answer can be found in search result [4]. To adjust the sag on your Super Deluxe Coil, you will need to install a \
coil spring and spring retainer, followed by some additional adjustments.

"""

    question = request.json.get("question")
    answer = request.json.get("answer")
    prompt = f"""

    Current question and generated answer:
    {question}

    Answer:
    {answer}
        """
    messages = [
        {"role": "system", "content": summarize_prompt},
        {"role": "user", "content": prompt},
    ]
    open_ai_result = make_openai_call(messages)
    cleaned_answer = open_ai_result.choices[0].message.content
    return cleaned_answer


@app.route("/aryn/interpret_os_result", methods=["POST", "OPTIONS"])
def interpret_os_result():
    if request.method == "OPTIONS":
        return optionsResp("POST")

    summarize_prompt = """
The following is a result from an opensearch query which was correctly generated to answer a user question. Given a \
user question and this result, synthesize an answer
doc_count does not mean number of crashes but instead number of references in the indexed documents, do not reference \
the doc_count field.

"""

    question = request.json.get("question")
    os_result = request.json.get("os_result")
    prompt = f"""

    User question:
    {question}

    OpenSearch Query result:
    {os_result}
        """
    messages = [
        {"role": "system", "content": summarize_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        open_ai_result = make_openai_call(messages)
        cleaned_answer = open_ai_result.choices[0].message.content
    except openai.error.InvalidRequestError as e:
        print(e)
        if "This model's maximum context length is" in e.args[0]:
            return "Unable to summarize result from OpenSearch due to content size."

    return cleaned_answer


@app.route(
    "/opensearch/<path:os_path>",
    methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
)
def proxy_opensearch(os_path):
    if request.method == "OPTIONS":
        return optionsResp("GET, POST, PUT, DELETE, HEAD")

    # log = request.method + " " + OPENSEARCH_URL + os_path
    # if request.is_json and request.content_length is not None:
    #     print(request.json)
    #     log += " " + str(request.json)

    url = OPENSEARCH_URL + os_path
    data = None if request.content_length is None else request.get_data()
    response = requests.request(
        method=request.method,
        params=request.args,
        url=url,
        data=data,
        headers=request.headers,
        verify=False,
    )
    # qa_logger.info(log)
    # qa_logger.info(str(response.json()))

    return response.json()


@app.route('/aryn/anthropic_rag', methods=['POST', 'OPTIONS'])
def anthropic_rag():
    if request.method == 'OPTIONS':
        return optionsResp('POST')

    question = request.json.get('question')
    os_result = request.json.get('os_result')

    user_prompt = """
    Search results: 
    """
    for i, s in enumerate(os_result["hits"]["hits"][0:10]):
        doc = ""
        doc += "<document>\n"
        doc += "Search result: " + str(i+1) + "\n"
        doc += s["_source"]["text_representation"] + "\n"
        doc += "</document>\n"
        user_prompt += doc + "\n"
    
    user_prompt += "<question>Question: " + question + " </question>"
    messages = [
      {"role": "user", "content": user_prompt}
    ]
    result = anthropic_client.messages.create(
        # model="claude-3-opus-20240229",
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        system=ANTHROPIC_RAG_PROMPT,
        messages=messages
    )
    
    return result.content[0].text


@app.route("/opensearch-version", methods=["GET", "OPTIONS"])
def opensearch_version(retries=3):
    if request.method == "OPTIONS":
        return optionsResp("GET")
    try:
        response = requests.request(
            method="GET",
            url=OPENSEARCH_URL,
            verify=False,
        )
        return response.json()["version"]["number"], 200
    except Exception as e:
        if retries <= 0:
            logger.error(f"OpenSearch not standing at {OPENSEARCH_URL}. Out of retries. Final error {e}")
            return "OpenSearch not found", 503
        logger.warning(
            f"OpenSearch not standing at {OPENSEARCH_URL}. Retrying in 1 sec. {retries-1} retries left. error {e}"
        )
        time.sleep(1)
        return opensearch_version(retries=retries - 1)


@app.route("/", methods=["GET", "OPTIONS"])
@app.route("/manifest.json", methods=["GET", "OPTIONS"])
@app.route("/static/<path:arg>", methods=["GET", "OPTIONS"])
@app.route("/viewPdf", methods=["GET", "OPTIONS"])
@app.route("/favicon.ico", methods=["GET", "OPTIONS"])
@app.route("/<arg>.png", methods=["GET", "OPTIONS"])
@app.route("/robots.txt", methods=["GET", "OPTIONS"])
def proxy_ui(arg=None):
    if request.method == "OPTIONS":
        return optionsResp("GET")

    resp = requests.request(
        method=request.method,
        params=request.args,
        url=UI_BASE + request.path,
        headers=request.headers,
        verify=False,
    )

    headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in badHeaders]

    return (resp.content, resp.status_code, headers)


@app.route("/healthz", methods=["GET"])
def healthz(arg=None):
    return "OK"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("PROXY %(message)s"))
    wsgilog = logging.getLogger("wsgi")
    wsgilog.propagate = False
    wsgilog.addHandler(sh)
    adapter = LoggingLogAdapter(wsgilog)

    # Use gevent WSGIServer for asynchronous behavior
    if os.environ.get("SSL", "1") == "0":
        logger.info("Proxy not serving over SSL.")
        http_server = WSGIServer(("0.0.0.0", PORT), app, log=adapter)
    else:
        http_server = WSGIServer(
            ("0.0.0.0", PORT),
            app,
            log=adapter,
            certfile=f"{HOST}-cert.pem",
            keyfile=f"{HOST}-key.pem",
        )
    logger.info(f"Serving on {PORT}...")
    http_server.serve_forever()
