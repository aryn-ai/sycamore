from PIL.Image import Image
import pdf2image
import pytest
from typing import cast
import warnings

import sycamore
from sycamore.data.table import Table, TableCell
from sycamore.data.element import create_element, TableElement
from sycamore.llms import LLM
from sycamore.llms.anthropic import Anthropic, AnthropicModels
from sycamore.llms.gemini import Gemini, GeminiModels
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.tests.config import TEST_DIR
from sycamore.tests.unit.data.test_table import SmithsonianSampleTable

from sycamore.transforms.partition import ArynPartitioner
from sycamore.transforms.table_structure.extract import VLMTableStructureExtractor


basic_table_path = TEST_DIR / "resources/data/pdfs/basic_table.pdf"


@pytest.fixture(scope="module")
def basic_table_image() -> Image:
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
