from typing import Optional
from pyarrow.filesystem import FileSystem
from pyarrow import fs
from sycamore.functions import HuggingFaceTokenizer
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.reader import DocSetReader
from sycamore.scans.file_scan import JsonManifestMetadataProvider
from sycamore.transforms.embed import SentenceTransformerEmbedder, OpenAIEmbedder
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.transforms.partition import SycamorePartitioner
import sycamore
from time import time
from pathlib import Path

from ray.data import ActorPoolStrategy

###########################

def get_fs():
    return fs.LocalFileSystem()


class ManifestReader(DocSetReader):
    def binary(
        self,
        binary_format: str,
        parallelism: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        metadata_provider: Optional[JsonManifestMetadataProvider] = None,
        file_range: Optional[list] = None,
        **resource_args
    ):
        paths = metadata_provider.get_paths()
        paths=paths if file_range == None else paths[file_range[0]:file_range[1]]
        return super().binary(
            paths=paths,
            binary_format=binary_format,
            parallelism=parallelism,
            filesystem=filesystem,
            metadata_provider=metadata_provider,
            **resource_args
        )

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
                  "dimension": 1536,
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

index = "metadata-eval-3.5-openai768"
s3_path = "s3://aryn-datasets-us-east-1/financebench/pdfs/"
manifest_path = "s3://aryn-datasets-us-east-1/financebench/manifest.json"
manifest_path_local="/home/admin/manifest.json"

# hf_model = "sentence-transformers/all-MiniLM-L6-v2"
# hf_model = "sentence-transformers/all-mpnet-base-v2"
hf_model = "Alibaba-NLP/gte-large-en-v1.5"
# hf_model = "FinLang/finance-embeddings-investopedia"

openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
tokenizer = HuggingFaceTokenizer(hf_model)

###########################

ctx = sycamore.init()
reader = ManifestReader(ctx)

# accel = 'cpu'
# cpus = 4
# if torch.cuda.is_available():
#     accel = 'cuda'
#     cpus = None
# elif torch.backends.mps.is_available():
#     accel = 'mps'

start = time()
# embedder = SentenceTransformerEmbedder(model_name=hf_model, batch_size=100)
embedder = OpenAIEmbedder(batch_size=100)

ds_list = []

# for i in [[0,15],[15,30],[30,45],[45,60],[60,75]]:
for i in [[30,45],[45,60],[60,61],[61,62],[62,63],[63,67],[67,75]]:
    ctx = sycamore.init()
    reader = ManifestReader(ctx)
    ds = (
        reader.binary(binary_format="pdf", metadata_provider=JsonManifestMetadataProvider(manifest_path_local), filesystem=get_fs(), file_range=i)
        .partition(partitioner=SycamorePartitioner(extract_table_structure=True, threshold=0.35), num_gpus=0.1, compute=ActorPoolStrategy(size=1))
        .regex_replace(COALESCE_WHITESPACE)
        .merge(merger=GreedyTextElementMerger(tokenizer, 768))
        .spread_properties(["path", "company", "year", "doc-type"])
        .explode()
        .embed(embedder=embedder, num_gpus=0.1)
    )
    ds_list.append(ds)

end = time()
print(f"Took {(end - start) / 60} mins")

###########################

for ds in ds_list:
    start = time()

    ds.write.opensearch(
        os_client_args=os_client_args,
        index_name=index,
        index_settings=index_settings,
    )

    end = time()
    print(f"Took {(end - start) / 60} mins")