import sycamore
from sycamore.data import Document, Element
from sycamore.docset import DocSet
from sycamore.functions import HuggingFaceTokenizer
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.utils.opensearch import guess_opensearch_host

import argparse
import random
from opensearchpy import OpenSearch
from typing import List, Dict, Optional


def make_element_from_text(text:str) -> Element: 

    return Element(text_representation=text)
#    return {'properties': {}, 'text_representation': text}

def generate_elements(keyword_to_probability:Dict[str, float], max_elements:int) -> list:

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
        # choose keywords to include based on the probability of each keyword
        selected_keywords = " ".join([k for k in keyword_to_probability if random.random() < keyword_to_probability[k]])
        strings.append(make_element_from_text(text+selected_keywords))
    return strings

def generate_doc(doc_id:int, keyword_to_probability:Dict[str, float], maxelems:int) -> Document:
    """
    Generate a document with random elements.

    Args:
        keywords (list): List of keywords.
        probabilities (list): List of probabilities corresponding to the keywords.
        maxelems (int): Maximum number of elements in each document.

    Returns:
        A Document objects with a random set of elements.
    """
    return Document (doc_id=doc_id, elements=generate_elements(keyword_to_probability, maxelems))
#    return {'doc_id':doc_id, 'elements':generate_elements(keywords, probabilities, maxelems)}


def generate_docset(keyword_to_probability:Dict[str, float], numdocs:int, maxelems:int) -> List[Document]:
    """
    Generate a document set with random elements.

    Args:
        keywords (list): List of keywords.
        probabilities (list): List of probabilities corresponding to the keywords.
        numdocs (int): Maximum number of documents in the docset.
        maxelems (int): Maximum number of elements in each document.

    Returns:
        list: A list of Document objects, each containing a random set of elements.
    """
    docset = []
    for i in range(numdocs):
        doc = generate_doc(str(i), keyword_to_probability, maxelems)
        docset.append(doc)
    return docset

def main():
    argparser = argparse.ArgumentParser(prog="synth_data_llm_filter")
    argparser.add_argument("--oshost", default=None, help="The OpenSearch host to use. Defaults to guessing based on whether it is in a container.")
    argparser.add_argument("--osport", default=9200, help="The OpenSearch port to use")
    argparser.add_argument("--numdocs", default=10, help="Number of documents to generate")
    argparser.add_argument("--maxelems", default=5, help="Maximum number of elements in each document")
    argparser.add_argument("--index", default=None, help="The OpenSearch index name to populate")
    args = argparser.parse_args()

    # The OpenSearch index name to populate.
    if args.index is not None:
        INDEX = args.index
    else:
        INDEX = "synth_10"

    if args.oshost is not None:
        opensearch_host = args.oshost
    else:
        opensearch_host = guess_opensearch_host()

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

    os_client = OpenSearch(**os_client_args)

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
    numdocs = int(args.numdocs)

    # The maximum number of elements in each document.
    maxelems = int(args.maxelems)


    keywords = ['cat', 'dog']
    probabilities = [0.1, 0.2]
    keyword_to_probability = {
    'cat': 0.1,
    'dog': 0.2,
    }
    docset = generate_docset(keyword_to_probability, numdocs, maxelems)

    print(f"Generated docset containing {len(docset)} documents")
    print(docset[0])

    context = sycamore.init()
    tokenizer = HuggingFaceTokenizer("thenlper/gte-small")
    llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)

    docset_loaded = (context.read.document(docs=docset)
                    .explode()
                    .write.opensearch(
                        os_client_args=os_client_args,
                        index_name=INDEX,
                        index_settings=index_settings,
                    )
    )


if __name__ == "__main__":
    exit(main())