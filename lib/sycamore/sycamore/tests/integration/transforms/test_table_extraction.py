from PIL import Image
import pdf2image
import pytest
from typing import cast
import warnings

import sycamore
from sycamore.data.bbox import BoundingBox
from sycamore.data.table import Table, TableCell
from sycamore.data.element import create_element, TableElement
from sycamore.llms import LLM
from sycamore.llms.anthropic import Anthropic, AnthropicModels
from sycamore.llms.chained_llm import ChainedLLM
from sycamore.llms.gemini import Gemini, GeminiModels
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.tests.config import TEST_DIR
from sycamore.tests.unit.data.test_table import SmithsonianSampleTable

from sycamore.transforms.partition import ArynPartitioner
from sycamore.transforms.table_structure.extract import TableTransformerStructureExtractor, VLMTableStructureExtractor


basic_table_path = TEST_DIR / "resources/data/pdfs/basic_table.pdf"


@pytest.fixture(scope="module")
def basic_table_image() -> Image.Image:
    images = pdf2image.convert_from_path(basic_table_path)
    return images[0]


@pytest.fixture(scope="function")
def basic_table_element() -> TableElement:
    return cast(
        TableElement,
        create_element(
            **{  # type: ignore
                "type": "table",
                "bbox": [0.11353424072265625, 0.09198168667879972, 0.9460228774126839, 0.33391876220703126],
            }
        ),
    )


def _compare_cells(table: Table) -> bool:
    new_cells = []
    for cell in table.cells:
        cell_dict = cell.to_dict()
        cell_dict["content"] = cell_dict["content"].replace("²", "2")
        new_cells.append(TableCell.from_dict(cell_dict))
    return new_cells == SmithsonianSampleTable().table().cells


def _check_llm(llm: LLM, basic_table_element, basic_table_image) -> None:
    extractor = VLMTableStructureExtractor(llm)
    new_elem = extractor.extract(element=basic_table_element, doc_image=basic_table_image)
    assert new_elem.table is not None

    print(new_elem.table.to_html())

    assert len(new_elem.table.cells) == 25

    # Note: The LLM has done a good job of extracting the table, but I hate to
    # assert on something so potentially non-deterministic. For now we just
    # warn if there is a mismatch.
    if not _compare_cells(new_elem.table):
        warnings.warn(f"Tables did not match for {llm.__class__.__name__}")


def test_gemini_table_structure_extractor(basic_table_element, basic_table_image):
    gemini = Gemini(GeminiModels.GEMINI_2_5_FLASH_PREVIEW, default_llm_kwargs={"max_output_tokens": 2048})
    _check_llm(gemini, basic_table_element, basic_table_image)


@pytest.mark.skip(reason="Not getting great results from Anthropic for this table")
def test_anthropic_table_structure_extractor(basic_table_element, basic_table_image):
    anthropic = Anthropic(AnthropicModels.CLAUDE_3_5_SONNET)
    _check_llm(anthropic, basic_table_element, basic_table_image)


def test_openai_table_structure_extractor(basic_table_element, basic_table_image):
    openai = OpenAI(OpenAIModels.GPT_4O_MINI)
    _check_llm(openai, basic_table_element, basic_table_image)


def test_chained_llm_table_structure_extractor(basic_table_element, basic_table_image):
    openai = OpenAI(OpenAIModels.GPT_4O_MINI)
    gemini = Gemini(GeminiModels.GEMINI_2_5_FLASH_PREVIEW, default_llm_kwargs={"max_output_tokens": 2048})
    chained_llm = ChainedLLM([openai, gemini], model_name="chained_llm_table_extractor")
    _check_llm(chained_llm, basic_table_element, basic_table_image)


def test_gemini_table_structure_extractor_from_sycamore():
    gemini = Gemini(GeminiModels.GEMINI_2_5_FLASH_PREVIEW, default_llm_kwargs={"max_output_tokens": 2048})
    extractor = VLMTableStructureExtractor(gemini)

    context = sycamore.init()
    docs = (
        context.read.binary(paths=[str(basic_table_path)], binary_format="pdf")
        .partition(
            partitioner=ArynPartitioner(
                use_partitioning_service=False,
                use_cache=False,
                extract_table_structure=True,
                table_structure_extractor=extractor,
            )
        )
        .filter_elements(lambda d: d.type == "table")
        .take_all()
    )

    assert len(docs) == 1
    assert len(docs[0].elements) == 1

    elem = docs[0].elements[0]
    assert isinstance(elem, TableElement)

    assert elem.table is not None
    assert len(elem.table.cells) == 25


def vtext(s: str, left: float, top: float, right: float, bot: float) -> dict:
    return {
        "text": s,
        "bbox": BoundingBox(left, top, right, bot),
        "vector": 1j,
    }


def test_tatr_extraction_vert() -> None:
    imgpath = TEST_DIR / "resources/data/imgs/vert_table.png"
    toks = [
        vtext("$9.99", 0.20611437908496733, 0.1766414141414141, 0.22572222222222227, 0.21073232323232322),
        vtext("Real estate", 0.20611437908496733, 0.23678535353535352, 0.22572222222222227, 0.3028409090909091),
        vtext("Osgiliath", 0.20611437908496733, 0.32891414141414144, 0.22572222222222227, 0.38446969696969696),
        vtext("$2.00", 0.1708202614379085, 0.1766414141414141, 0.1904281045751634, 0.21073232323232322),
        vtext("$1.00", 0.13552614379084965, 0.1766414141414141, 0.15513398692810457, 0.21073232323232322),
        vtext("Zoom Bot", 0.1708202614379085, 0.2410050505050506, 0.1904281045751634, 0.3028409090909091),
        vtext("Antarctica", 0.1708202614379085, 0.32138131313131313, 0.1904281045751634, 0.38446969696969696),
        vtext("Foo-bar", 0.13552614379084965, 0.25486616161616166, 0.15513398692810457, 0.3028409090909091),
        vtext("Fa la la la la", 0.13552614379084965, 0.310459595959596, 0.15513398692810457, 0.38446969696969696),
        vtext("Price:", 0.10023202614379084, 0.17664646464646463, 0.11983986928104574, 0.21534090909090903),
        vtext("Description:", 0.10023202614379084, 0.22292171717171713, 0.11983986928104574, 0.3028409090909091),
        vtext("Name:", 0.10023202614379084, 0.3415454545454546, 0.11983986928104574, 0.38446969696969696),
    ]
    elem = TableElement(tokens=toks)
    elem.bbox = BoundingBox(0.09090403837316176, 0.17140846946022725, 0.23596388872931987, 0.390420615456321)
    te = TableTransformerStructureExtractor()
    with Image.open(imgpath) as img:
        elem = te.extract(elem, img)
    assert elem.table
    csv = elem.table.to_csv()
    assert (
        csv
        == """Name:,Description:,Price:
Fa la la la la,Foo-bar,$1.00
Antarctica,Zoom Bot,$2.00
Osgiliath,Real estate,$9.99
"""
    )
