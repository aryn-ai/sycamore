from typing import TYPE_CHECKING

import pytest
import sycamore
from sycamore.tests.config import TEST_DIR

if TYPE_CHECKING:
    from sycamore.docset import DocSet


@pytest.fixture(scope="session")
def materialize_dir(tmp_path_factory):
    dir = tmp_path_factory.mktemp("materialize_dir")
    return dir


@pytest.fixture(scope="session")
def shared_ctx():
    ctx = sycamore.init()
    return ctx


@pytest.fixture(scope="session")
def partitioned_transformer_paper(shared_ctx, materialize_dir) -> "DocSet":
    from sycamore.transforms.partition import ArynPartitioner

    paths = str(TEST_DIR / "resources/data/pdfs/Transformer.pdf")

    ds = (
        shared_ctx.read.binary(paths, binary_format="pdf")
        .partition(partitioner=ArynPartitioner())
        .materialize(path=materialize_dir / "partitioned", source_mode=sycamore.MATERIALIZE_USE_STORED)
    )

    return ds


@pytest.fixture(scope="session")
def embedded_transformer_paper(partitioned_transformer_paper, materialize_dir) -> "DocSet":
    from sycamore.functions.tokenizer import HuggingFaceTokenizer
    from sycamore.transforms import COALESCE_WHITESPACE
    from sycamore.transforms.merge_elements import MarkedMerger
    from sycamore.transforms.embed import SentenceTransformerEmbedder

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = HuggingFaceTokenizer(model_name)

    ds = (
        partitioned_transformer_paper.regex_replace(COALESCE_WHITESPACE)
        .mark_bbox_preset(tokenizer=tokenizer)
        .merge(merger=MarkedMerger())
        .spread_properties(["path"])
        .split_elements(tokenizer=tokenizer, max_tokens=512)
        .explode()
        .embed(embedder=SentenceTransformerEmbedder(model_name=model_name, batch_size=100))
        .sketch(window=17)
        .materialize(path=materialize_dir / "embedded", source_mode=sycamore.MATERIALIZE_USE_STORED)
    )

    return ds
