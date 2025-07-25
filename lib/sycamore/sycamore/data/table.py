from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union, List, Sequence, TYPE_CHECKING
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

from sycamore.data.bbox import BoundingBox
from sycamore.utils.import_utils import requires_modules

if TYPE_CHECKING:
    from pandas import DataFrame
    from bs4 import Tag


# This is part of itertools in 3.10+.
# Adding here to support 3.9
def _pairwise(iterable):
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


# This data model is similar to that used by Textract and TableTransformers.
# It is intended to be a common intermediate representation for a variety of
# table structure recognition models.
@dataclass(frozen=True)
class TableCell:
    """Represents a single cell of a table.

    A cell can span multiple rows and columns, and can have an optional bounding box.
    """

    content: str
    rows: list[int]
    cols: list[int]
    is_header: bool = False
    bbox: Optional[BoundingBox] = None
    # Model/format specific properties
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.rows.sort()
        self.cols.sort()

        # Validate that row and column spans are contiguous.
        for a, b in _pairwise(self.rows):
            if a + 1 != b:
                raise ValueError(f"Found non-contiguous rows in {self}.")

        for a, b in _pairwise(self.cols):
            if a + 1 != b:
                raise ValueError(f"Found non-contiguous cols in {self}.")

    @classmethod
    def from_dict(cls, dict_obj: dict[str, Any]) -> "TableCell":
        for key in ["content", "rows", "cols"]:
            if key not in dict_obj:
                raise ValueError(f"Key {key} required to deserialize TableCell object.")

        kwargs = {"content": dict_obj["content"], "rows": dict_obj["rows"], "cols": dict_obj["cols"]}

        if "is_header" in dict_obj:
            kwargs["is_header"] = dict_obj["is_header"]

        if "bbox" in dict_obj and dict_obj["bbox"] is not None:
            kwargs["bbox"] = BoundingBox(**dict_obj["bbox"])

        if "properties" in dict_obj:
            kwargs["properties"] = dict_obj["properties"]

        return TableCell(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        dict_obj: dict[str, Union[Sequence[object], bool, dict]] = {
            "content": self.content,
            "rows": self.rows,
            "cols": self.cols,
        }

        if self.is_header:
            dict_obj["is_header"] = self.is_header

        if self.bbox:
            dict_obj["bbox"] = self.bbox.to_dict()

        if self.properties:
            dict_obj["properties"] = self.properties

        return dict_obj


DEFAULT_HTML_STYLE = """
table, th, td {
    border: solid thin;
    border-spacing: 2px;
    border-collapse: collapse;
    margin: 1.25em 0;
    padding: 0.5em
}
th {
    background-color: LightGray;
}
"""


class Table:
    """Represents a table from a document.

    This attempts to be a general representation that can represent a wide-variety of tables, including
    those with cells spanning multiple rows and columns, and complex multi-row headers. The table is
    represented as a simple list of cells, sorted by the minimum row, and column for that cell. This
    mimics common representations used by table extraction tools such as Amazon Textract and TableTransformers.

    Methods are provided to convert to common formats such as pandas, csv, and html. Some of these conversions
    are lossy, since, for instance, CSV does not natively support spanning cells.
    """

    def __init__(
        self, cells: list[TableCell], caption: Optional[str] = None, column_headers: Optional[list[str]] = None
    ):
        """Creates a new Table.

        Args:
            cells: The list of TableCells that make up this table.
            caption: An optional caption for this table.
        """

        self.cells: list[TableCell] = sorted(cells, key=lambda tc: (min(tc.rows), min(tc.cols)))
        self.caption = caption
        self.num_rows = max(max(c.rows) for c in self.cells) + 1
        self.num_cols = max(max(c.cols) for c in self.cells) + 1

        if column_headers is not None:
            self.column_headers = column_headers
        else:
            self.column_headers = self.to_pandas(column_header_only=True)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False

        if self.num_rows != other.num_rows or self.num_cols != other.num_cols:
            return False

        if self.cells != other.cells:
            return False

        return True

    def __hash__(self):
        return hash((self.cells))

    @classmethod
    def from_dict(cls, dict_obj: dict[str, Any]) -> "Table":
        """Construct a table from a dict representation."""
        if "cells" not in dict_obj:
            raise ValueError("Table dict must contain 'cells' key.")

        cells = [TableCell.from_dict(c) for c in dict_obj["cells"]]
        caption = dict_obj["caption"] if "caption" in dict_obj else None
        col_headers = dict_obj["column_headers"] if "column_headers" in dict_obj else None

        ret = Table(cells, caption, col_headers)

        if "num_rows" in dict_obj:
            assert ret.num_rows == dict_obj["num_rows"]

        if "num_cols" in dict_obj:
            assert ret.num_cols == dict_obj["num_cols"]

        return ret

    def to_dict(self) -> dict[str, Any]:
        """Converts the table to a dict representation."""
        d: dict[str, Union[list[dict[str, Any]], str, list[str], int]] = dict()
        d["cells"] = [c.to_dict() for c in self.cells]
        if self.caption is not None:
            d["caption"] = self.caption
        if hasattr(self, "column_headers") and self.column_headers is not None:
            d["column_headers"] = self.column_headers
        d["num_rows"] = self.num_rows
        d["num_cols"] = self.num_cols
        return d

    @staticmethod
    def extract_table_block(html_str: str):
        """
        Extracts the first <table>...</table> block from the given HTML string.
        Returns the table block as a Tag.
        """
        from bs4 import BeautifulSoup

        parsed = BeautifulSoup(html_str, "html.parser")
        return parsed.find("table")

    # TODO: There are likely edge cases where this will break or lose information. Nested or non-contiguous
    # headers are one likely source of issues. We also don't support missing closing tags (which are allowed in
    # the spec) because html.parser doesn't handle them. If and when this becomes an issue, we can consider
    # moving to the html5lib parser.
    @classmethod
    def from_html(cls, html_str: Optional[str] = None, html_tag: Optional["Tag"] = None) -> "Table":
        """
        Constructs a Table object from a well-formated HTML table.

        Args:
          html_str: The html string to parse. Must be enclosed in <table></table> tags.
          html_tag: A BeatifulSoup tag corresponding to the table. One of html_str or html_tag must be set.
        """
        from bs4 import BeautifulSoup, Tag

        # TODO: This doesn't account for rowgroup/colgroup handling, which can get quite tricky.

        if (html_str is not None and html_tag is not None) or (html_str is None and html_tag is None):
            raise ValueError("Exactly one of html_str and html_tag must be specified.")
        root: Union[Tag, BeautifulSoup]
        if html_str is not None:
            table_tag = cls.extract_table_block(html_str)
            if table_tag is None:
                raise ValueError("html_str must be a valid html table enclosed in <table></table> tags.")

            root = table_tag
        elif html_tag is not None:
            if html_tag.name != "table":
                raise ValueError(f"html_tag must correspond to a valid <table> tag. Got {html_tag.name}")
            root = html_tag
        else:
            # Should be unreachable
            raise RuntimeError(f"Unable to process from_html parameters: html_str={html_str} html_tag={html_tag}")

        cur_col = 0
        cur_row = -1

        cells = []
        caption = None
        assert isinstance(root, Tag), "Expected root to be a Tag"
        # Traverse the tree of elements in a pre-order traversal.
        for tag in root.find_all(recursive=True):
            assert isinstance(tag, Tag), "Expected root to be a Tag"
            if tag.name == "tr":
                cur_row += 1  # TODO: Should this be based on rowspan?
                cur_col = 0

            elif tag.name in {"td", "th"}:
                # We allow a missing tr for the first row to handle the case when
                # they have a thead.
                if cur_row < 0:
                    cur_row += 1
                if rowspan_str := tag.attrs.get("rowspan", "1"):
                    assert isinstance(rowspan_str, str)  # For mypy
                    rowspan = int(rowspan_str)
                if colspan_str := tag.attrs.get("colspan", "1"):
                    assert isinstance(colspan_str, str)  # For mypy
                    colspan = int(colspan_str)

                content = tag.get_text()

                cells.append(
                    TableCell(
                        content=content,
                        rows=list(range(cur_row, cur_row + rowspan)),
                        cols=list(range(cur_col, cur_col + colspan)),
                        is_header=(tag.name == "th"),
                    )
                )

                cur_col += colspan

            elif tag.name == "caption":
                caption = tag.get_text()

        # Fix columns where rowspans should be inserted
        candidate_bumpers: dict[int, list[tuple[int, int]]] = {}  # dict{row->list[(after, by)]}
        for c in cells:
            # If there are candidates for bumping this cell, check 'em
            if c.rows[0] in candidate_bumpers:
                bumpers = candidate_bumpers[c.rows[0]]
                for after, by in bumpers:
                    if c.cols[0] >= after:
                        for i in range(len(c.cols)):
                            c.cols[i] += by
            # If this cell is in multiple rows, add it as a candidate to the next few
            if len(c.rows) > 1:
                for row in c.rows[1:]:
                    if row not in candidate_bumpers:
                        candidate_bumpers[row] = []
                    candidate_bumpers[row].append((c.cols[0], len(c.cols)))
                    candidate_bumpers[row].sort()

        return Table(cells, caption=caption)

    # This algorithm is modified from the TableTransformers code. The conversion to Pandas/CSV
    # is necessarily lossy since these formats requires tables to be square and don't support
    # cells spanning multiple rows/columns.
    #
    # One of the main decisions here is how to render cells that span multiple rows and columns,
    # since they are not nativey supported in CSV. The standard approaches are (1) to duplicate
    # the cell value in each row/column or (2) to put the value in the first row/column in the cell
    # and empty values in the other. TableTransformers do the former, The textractor library supports
    # both, but defaults to the latter. Since our primary use case is preparing context for LLMs,
    # we speculate that duplication may create confusion, so we default to only displaying a cells
    # content for the first row/column for which it is applicable. The exception is for header rows,
    # where we duplicate values to each columnn to ensure that every column has a fully qualified header.
    def to_pandas(self, column_header_only: bool = False) -> Union["DataFrame", List[str]]:
        """Returns this table as a Pandas DataFrame.

        For example, Suppose a cell spans row 2-3 and columns 4-5.
        """

        from pandas import DataFrame
        import numpy

        # Find all row nums containing cells marked as headers.
        header_rows = sorted(set((row_num for cell in self.cells for row_num in cell.rows if cell.is_header)))

        # Find the number of initial rows that are header rows. This currently treats rows that are
        # marked as headers but are not part of the prefix of rows in the table as regular (non-header)
        # rows. This is wrong in general, but necessary for conversion to simpler formats like CSV.
        # Note that we also don't handle the case where only part of a row is a header. If any cell in
        # the row is marked as a header, all cells are treated as such for this conversion.
        i = -1
        for i in range(len(header_rows)):
            if header_rows[i] != i:
                break

        max_header_prefix_row = i

        table_array = numpy.empty([self.num_rows, self.num_cols], dtype="object")
        if len(self.cells) > 0:
            for cell in self.cells:
                # We treat header cells that are not at the beginning of the table
                # as regular cells.
                if cell.is_header and cell.rows[0] <= max_header_prefix_row:
                    # Put the value in the first row and all the columns.
                    for col in cell.cols:
                        table_array[cell.rows[0], col] = cell.content

                    # Put an empty string in each of the other rows. We want the
                    # header included in every column, but we don't need it repeated
                    # multiple times for a single column.
                    for row in cell.rows[1:]:
                        for col in cell.cols:
                            table_array[row, col] = ""

                else:
                    if not column_header_only:
                        for row in cell.rows:
                            for col in cell.cols:
                                if row == cell.rows[0] and col == cell.cols[0]:
                                    table_array[row, col] = cell.content
                                else:
                                    table_array[row, col] = ""

        header = table_array[: max_header_prefix_row + 1, :]

        flattened_header = []

        for npcol in header.transpose():
            flattened_header.append(" | ".join(OrderedDict.fromkeys((c for c in npcol if c not in [None, ""]))))
        if column_header_only:
            return flattened_header
        df = DataFrame(
            table_array[max_header_prefix_row + 1 :, :],
            index=None,
            columns=flattened_header if max_header_prefix_row >= 0 else None,
        )
        return df

    def to_csv(self, **kwargs) -> str:
        """Converts this table to a csv string.

        This conversion is made via Pandas.

        Args:
            kwargs: Keyword arguments to pass to the pandas to_csv method.
        """

        from pandas import DataFrame

        has_header = any((row_num == 0 for cell in self.cells for row_num in cell.rows if cell.is_header))

        pandas_kwargs = {"index": False, "header": has_header}
        pandas_kwargs.update(kwargs)
        df = self.to_pandas(column_header_only=False)
        assert isinstance(df, DataFrame), "Expected `to_pandas` to return a DataFrame"
        return df.to_csv(**pandas_kwargs)

    def to_html(self, pretty=False, wrap_in_html=False, style=DEFAULT_HTML_STYLE):
        """Converts this table to an HTML string.

        Cells with is_header=True will be converted to th tags. Cells spanning
        multiple rows or columns will have the rowspan or colspan attributes,
        respectively.

        Args:
            pretty: If True, pretty-prints the html. Otherwise returns it as a single line string.
                 Default is False.
            wrap_in_html: If True, wraps the output as a complete html page with <html>, <head>, and <body> tags.
                 If False, returns just the <table> fragment. Default is False.
            style: The CSS style to use if wrap_in_html is True. This will be included inline in a <style> tag
                 in the HTML header. The default applies some basic formatting and sets <th> cells to
                 have a grey background.
        """

        if wrap_in_html:
            root = ET.Element("html")
            head = ET.SubElement(root, "head")
            style_tag = ET.SubElement(head, "style")
            style_tag.text = style
            body = ET.SubElement(root, "body")
            table = ET.SubElement(body, "table")
        else:
            table = ET.Element("table")
            root = table

        curr_row = -1
        row = None

        if self.caption is not None:
            caption_cell = ET.SubElement(table, "caption")
            caption_cell.text = self.caption

        # TODO: We should eventually put these in <thead> and <tbody> tags.
        for cell in self.cells:
            cell_attribs = {}

            rowspan = len(cell.rows)
            colspan = len(cell.cols)

            if rowspan > 1:
                cell_attribs["rowspan"] = str(rowspan)
            if colspan > 1:
                cell_attribs["colspan"] = str(colspan)

            if cell.rows[0] > curr_row:
                curr_row = cell.rows[0]
                row = ET.SubElement(table, "tr")

            tcell = ET.SubElement(row, "th" if cell.is_header else "td", attrib=cell_attribs)
            tcell.text = cell.content

        if pretty:
            ET.indent(root)

        return ET.tostring(root, encoding="unicode")

    def to_tree(self) -> "TableTree":
        root = TableTree(tag="table")
        if len(self.cells) == 0:
            return root

        curr_row = 0
        row = TableTree(tag="tr")
        root.children.append(row)

        # TODO: We should eventually put these in <thead> and <tbody> tags.
        for cell in self.cells:

            rowspan = len(cell.rows)
            colspan = len(cell.cols)

            if cell.rows[0] > curr_row:
                curr_row = cell.rows[0]
                row = TableTree(tag="tr")
                root.children.append(row)

            leaf_tag = "th" if cell.is_header else "td"
            tcell = TableTree(tag=leaf_tag, rowspan=rowspan, colspan=colspan, text=cell.content)
            row.children.append(tcell)

        return root

    U = TypeVar("U", bound=Union[Image.Image, ImageDraw.ImageDraw])

    # TODO: This currently assumes that the bounding rectangles are on the same page.
    def draw(self, target: U) -> U:
        """Draw the bounding boxes for this table on the specified Image.

        Args:
           target: An Image or ImageDraw objects on which to draw this table.
              If target is an Image, an ImageDraw object will be created.
        """
        from sycamore.utils.image_utils import try_draw_boxes

        return try_draw_boxes(target, self.cells, color_fn=lambda _: "red", text_fn=lambda _, i: None)


class TableTree:
    def __init__(
        self,
        tag: str,
        colspan: Optional[int] = None,
        rowspan: Optional[int] = None,
        text: Optional[str] = None,
        children: Optional[list["TableTree"]] = None,
    ):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.text = text
        if children is None:
            self.children = []
        else:
            self.children = children

    def bracket(self) -> str:
        """Return the bracket format of this tree, which is what apted expects."""

        if self.tag in {"td", "th"}:
            result = f'"tag": {self.tag}, "colspan": {self.colspan}, "rowspan": {self.rowspan}, "text": {self.text}'
        else:
            result = f'"tag": {self.tag}'
        result += "".join(child.bracket() for child in self.children)
        return "{{{}}}".format(result)

    def get_size(self) -> int:
        return 1 + sum(child.get_size() for child in self.children)

    def to_html(self):
        if self.text:
            assert len(self.children) == 0, f"Found text in a non leaf node??? {self.bracket()}"
            return f'<{self.tag} colspan="{self.colspan}" rowspan="{self.rowspan}">{self.text}</{self.tag}>'
        else:
            return f'<{self.tag}>{"".join(c.to_html() for c in self.children)}</{self.tag}>'


@requires_modules("apted", extra="eval")
def ted_score(table1: Table, table2: Table) -> float:
    """Computes the tree edit distance (TED) score between two Tables

    https://github.com/ibm-aur-nlp/PubTabNet/blob/7b03ef8f54f747fa3accf7b9354520a41b30ab40/src/metric.py

    Args:
        table1:
        table2:
    """
    from apted import APTED

    tt1 = table1.to_tree()
    tt2 = table2.to_tree()

    distance = APTED(tt1, tt2).compute_edit_distance()
    return 1.0 - float(distance) / max(tt1.get_size(), tt2.get_size(), 1)
