from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import pytest
import re

from sycamore.data.bbox import BoundingBox
from sycamore.data.table import Table, TableCell


class TableFormatTestCase(ABC):
    """Test case for conversion between table formats."""

    @abstractmethod
    def canonical_html(self) -> str:
        pass

    def other_html(self) -> list[str]:
        return []

    @abstractmethod
    def csv(self) -> str:
        pass

    @abstractmethod
    def table(self) -> Table:
        pass


class SimpleTable(TableFormatTestCase):
    def canonical_html(self) -> str:
        return """
        <table>
          <tr>
            <td>1</td>
            <td>2</td>
          </tr>
          <tr>
            <td>3</td>
            <td>4</td>
          </tr>
        </table>
        """

    def other_html(self) -> list[str]:
        return [
            """
            <table>
              <tbody>
                <tr>
                  <td>1</td>
                  <td>2</td>
                </tr>
                <tr>
                  <td>3</td>
                  <td>4</td>
                </tr>
              </tbody>
            </table>
            """,
            """
            <table frame="hsides">
              <tr>
                <td>1</td>
                <td>2</td>
              </tr>
              <tr>
                <td>3</td>
                <td>4</td>
              </tr>
            </table>
            """,
        ]

    def csv(self) -> str:
        return "1,2\n3,4"

    def table(self) -> Table:
        return Table(
            [
                TableCell(content="1", rows=[0], cols=[0]),
                TableCell(content="2", rows=[0], cols=[1]),
                TableCell(content="3", rows=[1], cols=[0]),
                TableCell(content="4", rows=[1], cols=[1]),
            ]
        )


class SimpleTableWithHeader(TableFormatTestCase):
    def canonical_html(self) -> str:
        return """
        <table>
          <tr>
            <th>head1</th>
            <th>head2</th>
          </tr>
          <tr>
            <td>3</td>
            <td>4</td>
          </tr>
        </table>
        """

    def other_html(self):
        return [
            """
            <table>
              <thead>
                <tr>
                  <th>head1</th>
                  <th>head2</th>
                </tr>
              </thead>
              <tr>
                <td>3</td>
                <td>4</td>
              </tr>
            </table>
            """,
            """
            <table>
              <thead>
                <th>head1</th>
                <th>head2</th>
              </thead>
              <tr>
                <td>3</td>
                <td>4</td>
              </tr>
            </table>
            """,
            """
            <table>
              <thead>
                <tr>
                  <th>head1</th>
                  <th>head2</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>3</td>
                  <td>4</td>
                </tr>
              </tbody>
            </table>
            """,
        ]

    def csv(self) -> str:
        return "head1,head2\n3,4"

    def table(self) -> Table:
        return Table(
            [
                TableCell(content="head1", rows=[0], cols=[0], is_header=True),
                TableCell(content="head2", rows=[0], cols=[1], is_header=True),
                TableCell(content="3", rows=[1], cols=[0], is_header=False),
                TableCell(content="4", rows=[1], cols=[1], is_header=False),
            ]
        )


class SimpleTableMultiColHeader(TableFormatTestCase):
    def canonical_html(self) -> str:
        return """
        <table>
          <tr>
            <th colspan="2">multi head</th>
            <th>head2</th>
          </tr>
          <tr>
            <td>1</td>
            <td>2</td>
            <td>3</td>
          </tr>
          <tr>
            <td>4</td>
            <td>5</td>
            <td>6</td>
          </tr>
        </table>
        """

    def csv(self) -> str:
        return "multi head,multi head,head2\n1,2,3\n4,5,6"

    def table(self) -> Table:
        return Table(
            [
                TableCell(content="multi head", rows=[0], cols=[0, 1], is_header=True),
                TableCell(content="head2", rows=[0], cols=[2], is_header=True),
                TableCell(content="1", rows=[1], cols=[0], is_header=False),
                TableCell(content="2", rows=[1], cols=[1], is_header=False),
                TableCell(content="3", rows=[1], cols=[2], is_header=False),
                TableCell(content="4", rows=[2], cols=[0], is_header=False),
                TableCell(content="5", rows=[2], cols=[1], is_header=False),
                TableCell(content="6", rows=[2], cols=[2], is_header=False),
            ]
        )


class SimpleTableMultiRowHeader(TableFormatTestCase):
    def canonical_html(self) -> str:
        return """
        <table>
          <tr>
            <th rowspan="2">multi head</th>
            <th>head2_1</th>
          </tr>
          <tr>
            <th>head2_2</th>
          </tr>
          <tr>
            <td>1</td>
            <td>2</td>
          </tr>
          <tr>
            <td>3</td>
            <td>4</td>
          </tr>
        </table>
        """

    def csv(self) -> str:
        return "multi head,head2_1 | head2_2\n1,2\n3,4"

    def table(self) -> Table:
        return Table(
            [
                TableCell(content="multi head", rows=[0, 1], cols=[0], is_header=True),
                TableCell(content="head2_1", rows=[0], cols=[1], is_header=True),
                TableCell(content="head2_2", rows=[1], cols=[1], is_header=True),
                TableCell(content="1", rows=[2], cols=[0], is_header=False),
                TableCell(content="2", rows=[2], cols=[1], is_header=False),
                TableCell(content="3", rows=[3], cols=[0], is_header=False),
                TableCell(content="4", rows=[3], cols=[1], is_header=False),
            ]
        )


class SimpleTableMultiRowColHeader(TableFormatTestCase):
    def canonical_html(self) -> str:
        return """
        <table>
          <tr>
            <th rowspan="2" colspan="2">multi head</th>
            <th>head2_1</th>
          </tr>
          <tr>
            <th>head2_2</th>
          </tr>
          <tr>
            <td>1</td>
            <td>2</td>
            <td>3</td>
          </tr>
          <tr>
            <td>4</td>
            <td>5</td>
            <td>6</td>
          </tr>
        </table>
        """

    def csv(self) -> str:
        return "multi head,multi head,head2_1 | head2_2\n1,2,3\n4,5,6"

    def table(self) -> Table:
        return Table(
            [
                TableCell(content="multi head", rows=[0, 1], cols=[0, 1], is_header=True),
                TableCell(content="head2_1", rows=[0], cols=[2], is_header=True),
                TableCell(content="head2_2", rows=[1], cols=[2], is_header=True),
                TableCell(content="1", rows=[2], cols=[0], is_header=False),
                TableCell(content="2", rows=[2], cols=[1], is_header=False),
                TableCell(content="3", rows=[2], cols=[2], is_header=False),
                TableCell(content="4", rows=[3], cols=[0], is_header=False),
                TableCell(content="5", rows=[3], cols=[1], is_header=False),
                TableCell(content="6", rows=[3], cols=[2], is_header=False),
            ]
        )


class SmithsonianSampleTable(TableFormatTestCase):
    def canonical_html(self) -> str:
        return """
        <table>
         <caption>Specification values: Steel, Castings,
         Ann. A.S.T.M. A27-16, Class B;* P max. 0.06; S max. 0.05.</caption>
         <tr>
          <th rowspan="2">Grade.</th>
          <th rowspan="2">Yield Point.</th>
          <th colspan="2">Ultimate tensile strength</th>
          <th rowspan="2">Per cent elong. 50.8 mm or 2 in.</th>
          <th rowspan="2">Per cent reduct. area.</th>
         </tr>
         <tr>
          <th>kg/mm2</th>
          <th>lb/in2</th>
         </tr>
         <tr>
          <td>Hard</td>
          <td>0.45 ultimate</td>
          <td>56.2</td>
          <td>80,000</td>
          <td>15</td>
          <td>20</td>
         </tr>
         <tr>
          <td>Medium</td>
          <td>0.45 ultimate</td>
          <td>49.2</td>
          <td>70,000</td>
          <td>18</td>
          <td>25</td>
         </tr>
         <tr>
          <td>Soft</td>
          <td>0.45 ultimate</td>
          <td>42.2</td>
          <td>60,000</td>
          <td>22</td>
          <td>30</td>
         </tr>
        </table>
        """

    def csv(self) -> str:
        # ruff: noqa: E501
        return """Grade.,Yield Point.,Ultimate tensile strength | kg/mm2,Ultimate tensile strength | lb/in2,Per cent elong. 50.8 mm or 2 in.,Per cent reduct. area.
Hard,0.45 ultimate,56.2,"80,000",15,20
Medium,0.45 ultimate,49.2,"70,000",18,25
Soft,0.45 ultimate,42.2,"60,000",22,30
        """

    def table(self) -> Table:
        return Table(
            [
                TableCell(content="Grade.", rows=[0, 1], cols=[0], is_header=True),
                TableCell(content="Yield Point.", rows=[0, 1], cols=[1], is_header=True),
                TableCell(content="Ultimate tensile strength", rows=[0], cols=[2, 3], is_header=True),
                TableCell(content="Per cent elong. 50.8 mm or 2 in.", rows=[0, 1], cols=[4], is_header=True),
                TableCell(content="Per cent reduct. area.", rows=[0, 1], cols=[5], is_header=True),
                TableCell(content="kg/mm2", rows=[1], cols=[2], is_header=True),
                TableCell(content="lb/in2", rows=[1], cols=[3], is_header=True),
                TableCell(content="Hard", rows=[2], cols=[0]),
                TableCell(content="0.45 ultimate", rows=[2], cols=[1]),
                TableCell(content="56.2", rows=[2], cols=[2]),
                TableCell(content="80,000", rows=[2], cols=[3]),
                TableCell(content="15", rows=[2], cols=[4]),
                TableCell(content="20", rows=[2], cols=[5]),
                TableCell(content="Medium", rows=[3], cols=[0]),
                TableCell(content="0.45 ultimate", rows=[3], cols=[1]),
                TableCell(content="49.2", rows=[3], cols=[2]),
                TableCell(content="70,000", rows=[3], cols=[3]),
                TableCell(content="18", rows=[3], cols=[4]),
                TableCell(content="25", rows=[3], cols=[5]),
                TableCell(content="Soft", rows=[4], cols=[0]),
                TableCell(content="0.45 ultimate", rows=[4], cols=[1]),
                TableCell(content="42.2", rows=[4], cols=[2]),
                TableCell(content="60,000", rows=[4], cols=[3]),
                TableCell(content="22", rows=[4], cols=[4]),
                TableCell(content="30", rows=[4], cols=[5]),
            ],
            caption="Specification values: Steel, Castings, Ann. A.S.T.M. A27-16, Class B;* P max. 0.06; S max. 0.05.",
        )


test_cases = [
    SimpleTable(),
    SimpleTableWithHeader(),
    SimpleTableMultiColHeader(),
    SimpleTableMultiRowHeader(),
    SimpleTableMultiRowColHeader(),
    SmithsonianSampleTable(),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_to_csv(test_case):
    actual = test_case.table().to_csv()
    # print(actual)
    # print(test_case.csv().strip())

    assert actual.strip() == test_case.csv().strip()


def _remove_whitespace(s):
    return re.sub(r"\s+", "", s)


@pytest.mark.parametrize("test_case", test_cases)
def test_to_html(test_case):
    actual = _remove_whitespace(test_case.table().to_html())
    expected = _remove_whitespace(test_case.canonical_html())
    parsed_actual = BeautifulSoup(actual, "html.parser")
    parsed_expected = BeautifulSoup(expected, "html.parser")

    # print(BeautifulSoup(test_case.table().to_html(), "html.parser").prettify())
    # print(parsed_expected.prettify())

    assert parsed_actual == parsed_expected


def test_tablecell_from_dict():
    test_dict = {
        "content": "Test content",
        "rows": [1, 2],
        "cols": [3, 4],
        "is_header": True,
        "bbox": {"x1": 2.0, "y1": 3.0, "x2": 1.0, "y2": 4.0},
        "properties": {"prop1": "val1"},
    }

    cell = TableCell.from_dict(test_dict)

    assert cell.content == test_dict["content"]
    assert cell.rows == test_dict["rows"]
    assert cell.cols == test_dict["cols"]
    assert cell.is_header == test_dict["is_header"]
    assert cell.bbox == BoundingBox(**test_dict["bbox"])
    assert cell.properties == test_dict["properties"]


def test_tablecell_from_dict_missing_content():
    test_dict = {
        "rows": [1, 2],
        "cols": [3, 4],
        "bbox": {"x1": 2.0, "y1": 3.0, "x2": 1.0, "y2": 4.0},
        "properties": {"prop1": "val1"},
    }

    with pytest.raises(ValueError):
        cell = TableCell.from_dict(test_dict)  # noqa: F841


def test_table_from_dict():
    table_cells = [
        TableCell(content="multi head", rows=[0, 1], cols=[0, 1], is_header=True),
        TableCell(content="head2_1", rows=[0], cols=[2], is_header=True),
        TableCell(content="head2_2", rows=[1], cols=[2], is_header=True),
        TableCell(content="1", rows=[2], cols=[0], is_header=False),
        TableCell(content="2", rows=[2], cols=[1], is_header=False),
    ]

    expected = Table(table_cells)

    json_cells = [
        {
            "content": tc.content,
            "rows": tc.rows,
            "cols": tc.cols,
            "is_header": tc.is_header,
            "bbox": None if tc.bbox is None else tc.bbox.to_dict(),
            "properties": tc.properties,
        }
        for tc in table_cells
    ]

    actual = Table.from_dict({"cells": json_cells})
    assert actual == expected


def test_table_from_dict_missing():
    table_cells = [
        TableCell(content="multi head", rows=[0, 1], cols=[0, 1], is_header=True),
        TableCell(content="head2_1", rows=[0], cols=[2], is_header=True),
        TableCell(content="head2_2", rows=[1], cols=[2], is_header=True),
        TableCell(content="1", rows=[2], cols=[0], is_header=False),
        TableCell(content="2", rows=[2], cols=[1], is_header=False),
    ]

    caption = "Test caption"
    expected = Table(table_cells, caption=caption)

    json_cells = [
        {
            "content": tc.content,
            "rows": tc.rows,
            "cols": tc.cols,
            "is_header": tc.is_header,
            "bbox": None if tc.bbox is None else tc.bbox.to_dict(),
            "properties": tc.properties,
        }
        for tc in table_cells
    ]

    actual = Table.from_dict({"cells": json_cells, "caption": caption})
    assert actual == expected


@pytest.mark.parametrize("test_case", test_cases)
def test_from_html(test_case):
    actual = Table.from_html(html_str=test_case.canonical_html())
    expected = test_case.table()
    assert actual == expected

    if hasattr(test_case, "other_html"):
        for other_html in test_case.other_html():
            actual = Table.from_html(html_str=other_html)
            assert actual == expected
