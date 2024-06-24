import sycamore
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import MarkedMerger
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.tests.config import TEST_DIR
from sycamore.utils.time_trace import ray_logging_setup
import duckdb


def test_to_duckdb():
    table_name = "duckdb_table"
    db_url = ":default:"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    paths = str(TEST_DIR / "resources/data/pdfs/")

    OpenAI(OpenAIModels.GPT_3_5_TURBO_INSTRUCT.value)
    tokenizer = HuggingFaceTokenizer(model_name)

    ctx = sycamore.init(ray_args={"runtime_env": {"worker_process_setup_hook": ray_logging_setup}})

    ds = (
        ctx.read.binary(paths, binary_format="pdf")
        .partition(partitioner=UnstructuredPdfPartitioner())
        .regex_replace(COALESCE_WHITESPACE)
        .mark_bbox_preset(tokenizer=tokenizer)
        .merge(merger=MarkedMerger())
        .spread_properties(["path"])
        .split_elements(tokenizer=tokenizer, max_tokens=512)
        .explode()
        .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
    )
    ds_count = ds.count()
    ds.write.duckdb(table_name=table_name, db_url=db_url)
    conn = duckdb.connect(database=db_url)
    duckdb_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
    assert ds_count == int(duckdb_count.fetchone()[0])
