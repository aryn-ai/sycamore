from sycamore.data.element import create_element, Element, ImageElement


def test_create_element_bad_type():
    e = create_element(type=None)
    assert isinstance(e, Element)

    e = create_element(type={})
    assert isinstance(e, Element)

    e = create_element(type="iMaGE")
    assert isinstance(e, ImageElement)
