import os
import uuid

import boto3
from urllib.parse import urlparse
from opensearchpy import OpenSearch

import sycamore
from sycamore.context import OpenSearchArgs
from sycamore.functions import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.utils.cache import S3Cache


def test_pdf_to_opensearch_with_llm_caching():
    os_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    index_settings = {
        "body": {
            "settings": {
                "index.knn": True,
                "number_of_shards": 5,
                "number_of_replicas": 1,
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {"name": "hnsw", "engine": "faiss"},
                    },
                    "text": {"type": "text"},
                }
            },
        }
    }

    # ruff: noqa: E501
    title_context_template = """
    ELEMENT 1: Jupiter's Moons
    ELEMENT 2: Ganymede 2020
    ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
    ELEMENT 4: From Wikipedia, the free encyclopedia
    ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
    =========
    "Ganymede 2020"

    ELEMENT 1: FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
    ELEMENT 2: Tarun Kalluri * UCSD
    ELEMENT 3: Deepak Pathak CMU
    ELEMENT 4: Manmohan Chandraker UCSD
    ELEMENT 5: Du Tran Facebook AI
    ELEMENT 6: https://tarun005.github.io/FLAVR/
    ELEMENT 7: 2 2 0 2
    ELEMENT 8: b e F 4 2
    ELEMENT 9: ]
    ELEMENT 10: V C . s c [
    ========
    "FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation"

    """

    author_context_template = """
        ELEMENT 1: Jupiter's Moons
        ELEMENT 2: Ganymede 2020
        ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
        ELEMENT 4: From Wikipedia, the free encyclopedia
        ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
        =========
        Audi Laupe, Serena K. Goldberg

        ELEMENT 1: FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
        ELEMENT 2: Tarun Kalluri * UCSD
        ELEMENT 3: Deepak Pathak CMU
        ELEMENT 4: Manmohan Chandraker UCSD
        ELEMENT 5: Du Tran Facebook AI
        ELEMENT 6: https://tarun005.github.io/FLAVR/
        ELEMENT 7: 2 2 0 2
        ELEMENT 8: b e F 4 2
        ELEMENT 9: ]
        ELEMENT 10: V C . s c [
        ========
        Tarun Kalluri, Deepak Pathak, Manmohan Chandraker, Du Tran

        """

    s3_cache_base_path = os.environ.get("SYCAMORE_S3_TEMP_PATH", "s3://aryn-sycamore-integ-temp/")
    test_path = str(uuid.uuid4())
    s3_cache_path = os.path.join(s3_cache_base_path, test_path)
    parsed_s3_url = urlparse(s3_cache_path)
    bucket = parsed_s3_url.netloc
    s3_path = parsed_s3_url.path

    pdf_base_path = TEST_DIR / "resources/data/pdfs/"
    paths = str(pdf_base_path)

    s3_client = boto3.client("s3")
    openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value, cache=S3Cache(s3_cache_path))
    tokenizer = HuggingFaceTokenizer("thenlper/gte-small")
    keys = set()

    try:
        context = sycamore.init(
            opensearch_args=OpenSearchArgs(os_client_args, "toyindex", index_settings),
            llm=openai_llm,
        )
        ds = (
            context.read.binary(paths, binary_format="pdf")
            .partition(partitioner=UnstructuredPdfPartitioner())
            .extract_entity(entity_extractor=OpenAIEntityExtractor("title", prompt_template=title_context_template))
            .extract_entity(entity_extractor=OpenAIEntityExtractor("authors", prompt_template=author_context_template))
            .merge(GreedyTextElementMerger(tokenizer=tokenizer, max_tokens=300))
            .explode()
            .sketch()
            .embed(
                embedder=SentenceTransformerEmbedder(
                    batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            )
        )
        ds.write.opensearch()

        OpenSearch(**os_client_args).indices.delete("toyindex")

        # validate caching

        if s3_path.startswith("/"):
            s3_path = s3_path.lstrip("/")
        if not s3_path.endswith("/"):
            s3_path = s3_path + "/"

        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_path)

        if "Contents" in response:
            for obj in response["Contents"]:
                keys.add(obj["Key"])

        # assert we've cached 2 (2 extract_entity calls) * number of pdfs
        assert len(keys) == 2 * len(list(pdf_base_path.glob("*.pdf")))
    finally:
        if len(keys) > 0:
            s3_client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys]})
        s3_client.delete_object(Bucket=bucket, Key=s3_path)
