import sys

import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import SycamorePartitioner
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.utils.time_trace import ray_logging_setup
from sycamore.tests.config import TEST_DIR
from simple_config import title_template

sys.path.append("../sycamore")

table_name = "duckdb_table"
db_url = "tmp.db"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
paths = str(TEST_DIR / "resources/data/pdfs/")
davinci_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)

tokenizer = HuggingFaceTokenizer(model_name)

ctx = sycamore.init(ray_args={"runtime_env": {"worker_process_setup_hook": ray_logging_setup}})

ds = (
    ctx.read.binary(paths, binary_format="pdf")
    .partition(partitioner=SycamorePartitioner())
    .regex_replace(COALESCE_WHITESPACE)
    .extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=davinci_llm, prompt_template=title_template))
    .mark_bbox_preset(tokenizer=tokenizer)
    .merge(merger=MarkedMerger())
    .spread_properties(["path"])
    .split_elements(tokenizer=tokenizer, max_tokens=512)
    .explode()
    .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
)
ds_count = ds.count()
ds.write.duckdb(table_name=table_name, db_url=db_url)
