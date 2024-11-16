from sycamore.data import Document, Element, Table, TableCell
from sycamore.llms import OpenAI
from sycamore.transforms.extract_table_properties import ExtractTableProperties
from PIL import Image
from io import BytesIO


class TestExtractTableProperties:
    def test_extract_key_value_pair(self, mocker):
        table_cells = [
            TableCell(content="head1", rows=[0], cols=[0], is_header=True),
            TableCell(content="head2", rows=[0], cols=[1], is_header=True),
            TableCell(content="key1", rows=[1], cols=[0], is_header=False),
            TableCell(content="val1", rows=[1], cols=[1], is_header=False),
        ]
        table = Table(table_cells)
        table_bbox = (0.1, 0.2, 0.8, 0.9)
        table_element = Element(
            type="table",
            bbox=table_bbox,
            properties={"page_number": 1, "title": {"rows": None, "columns": None}},
            table=table,
            tokens=None,
        )

        self.doc = Document(
            doc_id="doc_id",
            type="pdf",
            text_representation="text_representation",
            elements=[table_element],
            properties={"int": 0, "float": 3.14, "list": [1, 2, 3, 4], "tuple": (1, "tuple")},
            binary_representation=b"<dummy>",
        )

        mock_split_and_convert_to_image = mocker.patch(
            "sycamore.transforms.extract_table_properties.split_and_convert_to_image"
        )
        image = Image.new("RGB", (100, 100), color="white")
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_data = img_byte_arr.getvalue()

        mock_image = Document()
        mock_image.binary_representation = img_data
        mock_image.properties = {
            "size": (100, 100),
            "mode": "RGB",
        }
        mock_image.binary_representation = img_data

        mock_split_and_convert_to_image.return_value = [mock_image]

        mock_frombytes = mocker.patch("PIL.Image.frombytes")
        mock_frombytes.return_value = image

        llm = mocker.Mock(spec=OpenAI)
        llm.generate.return_value = '{"key1":"val1"}'
        llm.format_image.return_value = {"type": "image", "data": "dummy"}

        property_name = "llm_response"
        doc1 = ExtractTableProperties(None, parameters=[property_name, llm]).run(self.doc)

        assert (doc1.elements[0].properties.get("llm_response")) == {"key1": "val1"}
