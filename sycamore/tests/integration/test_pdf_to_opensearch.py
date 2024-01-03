import sycamore
from sycamore.transforms.embed import SentenceTransformerEmbedder

from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.tests.config import TEST_DIR


def test_pdf_to_opensearch():
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
                    "embeddings": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {"name": "hnsw", "engine": "nmslib"},
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

    paths = str(TEST_DIR / "resources/data/pdfs/")

    openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)

    context = sycamore.init()
    ds = (
        context.read.binary(paths, binary_format="pdf")
        .partition(partitioner=UnstructuredPdfPartitioner())
        .extract_entity(
            entity_extractor=OpenAIEntityExtractor("title", llm=openai_llm, prompt_template=title_context_template)
        )
        .extract_entity(
            entity_extractor=OpenAIEntityExtractor("authors", llm=openai_llm, prompt_template=author_context_template)
        )
        .explode()
        .embed(
            embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
    )

    ds.write.opensearch(
        os_client_args=os_client_args,
        index_name="toyindex",
        index_settings=index_settings,
    )
