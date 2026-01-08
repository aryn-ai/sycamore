from sycamore.schema import DataType
from sycamore.utils.predicates import PredicateExpressionParser


def test_parse():
    parser = PredicateExpressionParser()

    property_type = DataType.INT
    extracted_value = 2

    predicate1 = "x > 1"
    assert parser.evaluate(predicate1, property_type, extracted_value)

    predicate2 = "x > 2"
    assert not parser.evaluate(predicate2, property_type, extracted_value)

    predicate3 = "(x == 2)"
    assert parser.evaluate(predicate3, property_type, extracted_value)

    predicate4 = "(x >= 1) AND (x < 10)"
    assert parser.evaluate(predicate4, property_type, extracted_value)