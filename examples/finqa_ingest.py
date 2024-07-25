from pyarrow import fs
from sycamore.functions import HuggingFaceTokenizer
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.transforms.partition import SycamorePartitioner
import sycamore
from time import time

from ray.data import ActorPoolStrategy

###########################

os_client_args = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "http_compress": True,
    "http_auth": ('admin', 'admin'),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120
}

index_settings = {
    "body": {
        "settings": {
            "index.knn": True,
            "number_of_shards": 5,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "embedding": {
                  "dimension": 768,
                  "method": {
                    "engine": "faiss",
                    "space_type": "l2",
                    "name": "hnsw",
                    "parameters": {}
                  },
                  "type": "knn_vector"
                },
            }
        }
    }
}

index = "finqabench"
hf_model = "sentence-transformers/all-mpnet-base-v2"
tokenizer = HuggingFaceTokenizer(hf_model)
embedder = SentenceTransformerEmbedder(model_name=hf_model, batch_size=100)

start = time()
path = "/home/admin/pdfs/finqa/b4266e40-1de6-4a34-9dfb-8632b8bd57e0.pdf"

ctx = sycamore.init()
ds = (
    ctx.read.binary(path, binary_format="pdf", filter_paths_by_extension=False)
    .partition(partitioner=SycamorePartitioner(extract_table_structure=True, threshold=0.35, use_ocr=True), num_gpus=0.1, compute=ActorPoolStrategy(size=1))
    .regex_replace(COALESCE_WHITESPACE)
    .merge(merger=GreedyTextElementMerger(tokenizer, 512))
    .explode()
    .embed(embedder=embedder, num_gpus=0.1)
)

end = time()
print(f"Took {(end - start) / 60} mins")


start = time()

ds.write.opensearch(
    os_client_args=os_client_args,
    index_name=index,
    index_settings=index_settings,
)

end = time()
print(f"Took {(end - start) / 60} mins")
