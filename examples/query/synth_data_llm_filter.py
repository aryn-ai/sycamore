import sycamore
from sycamore.functions import HuggingFaceTokenizer
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms.embed import SentenceTransformerEmbedder
import os
from opensearchpy import OpenSearch
import argparse
import random
from sycamore.data import Document, Element
from sycamore.docset import DocSet

argparser = argparse.ArgumentParser(prog="synth_data_llm_filter")
argparser.add_argument("--oshost", default=None)
argparser.add_argument("--osport", default=9200)
argparser.add_argument("--maxdocs", default=10)
argparser.add_argument("--maxelems", default=5)
argparser.add_argument("--index", default=None)
args = argparser.parse_args()

# The OpenSearch index name to populate.
if args.index is not None:
    INDEX = args.index
else:
    INDEX = "synth_10"

# The OpenSearch instance to use.
if os.path.exists("/.dockerenv"):
    opensearch_host = "opensearch"
    print("Assuming we are in a Sycamore Jupyter container, using opensearch for OpenSearch host")
else:
    opensearch_host = "localhost"
    print("Assuming we are running outside of a container, using localhost for OpenSearch host")

opensearch_port = args.osport

os_client_args = {
    "hosts": [{"host": opensearch_host, "port": opensearch_port}],
    "http_compress": True,
    "http_auth": ("admin", "admin"),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120,
}

os_client = OpenSearch(**os_client_args)  # type: ignore

index_settings = {
    "body": {
        "settings": {"index.knn": True, "number_of_shards": 5, "number_of_replicas": 1},
        "mappings": {
            "properties": {
                "embedding": {
                    "dimension": 384,
                    "method": {
                        "engine": "faiss",
                        "space_type": "l2",
                        "name": "hnsw",
                        "parameters": {},
                    },
                    "type": "knn_vector",
                }
            }
        },
    }
}

# The number of documents to generate.
max_docs = int(args.maxdocs)

# The maximum number of elements in each document.
max_elems = int(args.maxelems)


def make_element(text):
    """
    Creates an element with the text representation.

    Args:
        text (str): The text representation of the element.

    Returns:
        dict: A dictionary representing the element with the following keys:
            - 'properties': A dictionary of properties associated with the element.
            - 'text_representation': The text representation of the element.
    """
#    return Element(text_representation=text)
    return {'properties': {}, 'text_representation': text}

def generate_elements(keywords, probabilities, max_elements):

    strings = []
    num = random.randint(1, max_elements)
    base_text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor in enim sit amet tristique. 
    In eu egestas ex. Maecenas porttitor gravida libero ac porttitor. Morbi elementum eleifend nisl.
     Nullam auctor magna nec mollis ultricies. Mauris non pulvinar turpis. Pellentesque dictum 
     tortor ac elit feugiat varius. Nulla sollicitudin, lectus non venenatis finibus, nulla ex 
     tempor urna, id ornare est nisl et lacus. Pellentesque commodo turpis in nibh aliquet viverra. 
     Curabitur varius, elit nec finibus scelerisque, nibh eros congue dui, et ultrices justo magna 
     eu sem. Sed ultricies posuere posuere. Ut euismod neque at diam interdum, sed rhoncus quam 
     iaculis. Morbi sit amet tempus orci, a dictum metus. Donec a imperdiet sem. Vestibulum tincidunt 
     gravida turpis vitae aliquet.
    """
    for i in range(num):
        text = "base_text" + str(i)
        selected_keywords = ""
        for keyword, probability in zip(keywords, probabilities):
            if random.random() < probability:
                selected_keywords = selected_keywords + " " + keyword
        strings.append(make_element(text+selected_keywords))
    return strings

def generate_doc(doc_id, keywords, probabilities, max_elems):
    """
    Generate a document with random elements.

    Args:
        keywords (list): List of keywords.
        probabilities (list): List of probabilities corresponding to the keywords.
        max_elems (int): Maximum number of elements in each document.

    Returns:
        A Document objects with a random set of elements.
    """
    return Document (doc_id=doc_id, elements=generate_elements(keywords, probabilities, max_elems))
#    return {'doc_id':doc_id, 'elements':generate_elements(keywords, probabilities, max_elems)}


def generate_docset(keywords, probabilities, max_docs, max_elems):
    """
    Generate a document set with random elements.

    Args:
        keywords (list): List of keywords.
        probabilities (list): List of probabilities corresponding to the keywords.
        max_docs (int): Maximum number of documents in the docset.
        max_elems (int): Maximum number of elements in each document.

    Returns:
        list: A list of Document objects, each containing a random set of elements.
    """
    docset = []
    for i in range(max_docs):
        doc = generate_doc(str(i), keywords, probabilities, max_elems)
        docset.append(doc)
    return docset


keywords = ['cat', 'dog']
probabilities = [0.1, 0.2]
docset = generate_docset(keywords, probabilities, max_docs, max_elems)

print(f"Generated docset containing {len(docset)} documents")
print(docset[0])

context = sycamore.init()
tokenizer = HuggingFaceTokenizer("thenlper/gte-small")
llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)

docset_loaded = (context.read.document(docs=docset)
                .explode())
# partitioning docset
docset_loaded.write.opensearch(
        os_client_args=os_client_args,
        index_name=INDEX,
        index_settings=index_settings,
    )
