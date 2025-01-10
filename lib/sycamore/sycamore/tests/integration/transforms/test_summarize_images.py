import sycamore
from sycamore.context import ExecMode
from sycamore.transforms.partition import ArynPartitioner
from sycamore.transforms.summarize_images import LLMImageSummarizer, OpenAIImageSummarizer, SummarizeImages
from sycamore.tests.config import TEST_DIR
from sycamore.llms.bedrock import BedrockModels, Bedrock
from sycamore.llms.anthropic import AnthropicModels, Anthropic


def test_summarize_images_openai():
    path = TEST_DIR / "resources/data/pdfs/Ray_page11.pdf"

    context = sycamore.init()
    image_docs = (
        context.read.binary(paths=[str(path)], binary_format="pdf")
        .partition(ArynPartitioner(extract_images=True, use_partitioning_service=False, use_cache=False))
        .transform(SummarizeImages, summarizer=OpenAIImageSummarizer())
        .explode()
        .filter(lambda d: d.type == "Image")
        .take_all()
    )

    assert len(image_docs) == 1
    assert image_docs[0].properties["summary"]["is_graph"]


def test_summarize_images_bedrock_claude():
    llm = Bedrock(BedrockModels.CLAUDE_3_5_SONNET)

    path = TEST_DIR / "resources/data/pdfs/Ray_page11.pdf"

    context = sycamore.init()
    image_docs = (
        context.read.binary(paths=[str(path)], binary_format="pdf")
        .partition(ArynPartitioner(extract_images=True, use_partitioning_service=False, use_cache=False))
        .transform(SummarizeImages, summarizer=LLMImageSummarizer(llm=llm))
        .explode()
        .filter(lambda d: d.type == "Image")
        .take_all()
    )

    assert len(image_docs) == 1
    assert image_docs[0].properties["summary"]["is_graph"]


def test_summarize_images_anthropic_claude():
    llm = Anthropic(AnthropicModels.CLAUDE_3_5_SONNET)

    path = TEST_DIR / "resources/data/pdfs/Ray_page11.pdf"

    context = sycamore.init(exec_mode=ExecMode.LOCAL)
    image_docs = (
        context.read.binary(paths=[str(path)], binary_format="pdf")
        .partition(ArynPartitioner(extract_images=True, use_partitioning_service=False, use_cache=False))
        .transform(SummarizeImages, summarizer=LLMImageSummarizer(llm=llm))
        .explode()
        .filter(lambda d: d.type == "Image")
        .take_all()
    )

    assert len(image_docs) == 1
    assert image_docs[0].properties["summary"]["is_graph"]
