from io import BytesIO
import msgpack

from sycamore.data.element import create_element, Element, ImageElement, TableElement
from sycamore.data.table import Table, TableCell


def test_create_element_bad_type():
    e = create_element(type=None)
    assert isinstance(e, Element)

    e = create_element(type={})
    assert isinstance(e, Element)

    e = create_element(type="iMaGE")
    assert isinstance(e, ImageElement)


def test_field_to_value_table():
    table = Table(
        [
            TableCell(content="head1", rows=[0], cols=[0], is_header=True),
            TableCell(content="head2", rows=[0], cols=[1], is_header=True),
            TableCell(content="3", rows=[1], cols=[0], is_header=False),
            TableCell(content="4", rows=[1], cols=[1], is_header=False),
        ]
    )

    elem = create_element(type="table", table=table, properties={"parent": {"child1": 1, "child2": 2}})
    assert isinstance(elem, TableElement)

    assert elem.field_to_value("properties.parent.child1") == 1
    assert elem.field_to_value("properties.parent.child2") == 2
    assert elem.field_to_value("properties.parent.child3") is None

    assert elem.field_to_value("text_representation") == "head1,head2\n3,4\n"


def test_web_serialize_deserialize():
    table = Table(
        [
            TableCell(content="head1", rows=[0], cols=[0], is_header=True),
            TableCell(content="head2", rows=[0], cols=[1], is_header=True),
            TableCell(content="3", rows=[1], cols=[0], is_header=False),
            TableCell(content="4", rows=[1], cols=[1], is_header=False),
        ]
    )
    expected_element = create_element(type="table", table=table, properties={"parent": {"child1": 1, "child2": 2}})
    bits = BytesIO()
    expected_element.web_serialize(bits)
    bits.seek(0)
    actual_element = create_element(**msgpack.unpackb(bits.read()))
    assert isinstance(actual_element, TableElement)
    assert actual_element.table == table
    assert actual_element.properties == expected_element.properties

    assert actual_element.field_to_value("properties.parent.child1") == 1
    assert actual_element.field_to_value("properties.parent.child2") == 2
    assert actual_element.field_to_value("properties.parent.child3") is None

    assert actual_element.field_to_value("text_representation") == "head1,head2\n3,4\n"
