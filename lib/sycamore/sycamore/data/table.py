from collections import OrderedDict
from dataclasses import dataclass, field
import itertools
from typing import Any, Optional
import xml.etree.ElementTree as ET

from sycamore.data import BoundingBox
from PIL import Image, ImageDraw
import numpy as np
from pandas import DataFrame


# This data model is similar to that used by Textract and TableTransformers.
# It is intended to be a common intermediate representation for a variety of
# table structure recognition models.
@dataclass(frozen=True)
class TableCell:
    content: str
    rows: list[int]
    cols: list[int]
    is_header: bool = False
    bbox: Optional[BoundingBox] = None
    # Model/format specific properties
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate that row and column spans are contiguous.
        for a, b in itertools.pairwise(self.rows):
            if a + 1 != b:
                raise ValueError(f"Found non-contiguous rows in {self}.")

        for a, b in itertools.pairwise(self.cols):
            if a + 1 != b:
                raise ValueError(f"Found non-contiguous cols in {self}.")


class Table:
    def __init__(self, cells: list[TableCell], caption: Optional[str] = None):
        self.cells: list[TableCell] = sorted(cells, key=lambda tc: (min(tc.rows), min(tc.cols)))
        self.caption = caption
        self.num_rows = max(max(c.rows) for c in self.cells) + 1
        self.num_cols = max(max(c.cols) for c in self.cells) + 1

    def __eq__(self, other):
        if type(other) is not type(self):
            return False

        if self.num_rows != other.num_rows or self.num_cols != other.num_cols:
            return False

        # TODO: Yikes! this is expensive. Do the sorting in init or something.
        if self.cells != other.cells:
            return False

        return True

    def __hash__(self):
        return hash((self.cells))

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
    def to_pandas(self) -> DataFrame:
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

        table_array = np.empty([self.num_rows, self.num_cols], dtype="object")
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
                    table_array[cell.rows[0], cell.cols[0]] = cell.content
                    for row in cell.rows[1:]:
                        for col in cell.cols[1:]:
                            table_array[row, col] = ""

        header = table_array[: max_header_prefix_row + 1, :]

        flattened_header = []

        for npcol in header.transpose():
            flattened_header.append(" | ".join(OrderedDict.fromkeys((c for c in npcol if c != ""))))

        df = DataFrame(
            table_array[max_header_prefix_row + 1 :, :],
            index=None,
            columns=flattened_header if max_header_prefix_row >= 0 else None,
        )
        return df

    def to_csv(self):
        has_header = any((row_num == 0 for cell in self.cells for row_num in cell.rows if cell.is_header))
        return self.to_pandas().to_csv(index=False, header=has_header)

    def to_html(self):
        table = ET.Element("table")

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

        return ET.tostring(table, encoding="unicode")

    # TODO: This currently assumes that the bounding rectangles are on the same page.
    def draw(self, image: Image) -> Image:
        """Draw the bounding boxes for this table on the specified Image."""
        width, height = image.size

        canvas = ImageDraw.Draw(image)

        for cell in self.cells:
            if cell.bbox is not None:
                coords = cell.bbox.to_absolute(width, height).coordinates
                canvas.rectangle(coords, outline="red")  # TODO color

        return image
