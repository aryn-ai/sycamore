import pytest

from sycamore.utils.predicates import PredicateExpressionParser


def test_parse_datatype_int():
    extracted_value = 2

    predicate1 = "x > 1"
    assert PredicateExpressionParser.evaluate(predicate1, extracted_value)

    predicate2 = "x > 2"
    assert not PredicateExpressionParser.evaluate(predicate2, extracted_value)

    predicate3 = "(x == 2)"
    assert PredicateExpressionParser.evaluate(predicate3, extracted_value)

    predicate4 = "(x >= 1) AND (x < 10)"
    assert PredicateExpressionParser.evaluate(predicate4, extracted_value)


def test_parse_datatype_bool():
    extracted_value = False
    predicate1 = "x == False"
    assert PredicateExpressionParser.evaluate(predicate1, extracted_value)

    extracted_value = False
    predicate2 = "(x == True) or (x == False)"
    assert PredicateExpressionParser.evaluate(predicate2, extracted_value)

    extracted_value = False
    predicate3 = "x != True"
    assert PredicateExpressionParser.evaluate(predicate3, extracted_value)


def test_parse_datatype_float():
    extracted_value = 10.0
    predicate1 = "x > 1.0"
    assert PredicateExpressionParser.evaluate(predicate1, extracted_value)


def test_parse_datatype_string():
    extracted_value = "abc"
    predicate1 = "x like 'abcd'"
    assert PredicateExpressionParser.evaluate(predicate1, extracted_value)

    extracted_value = "xyz"
    predicate2 = "x == 'xyz'"
    assert PredicateExpressionParser.evaluate(predicate2, extracted_value)

    extracted_value = "abc"
    predicate3 = "x != 'abcd'"
    assert PredicateExpressionParser.evaluate(predicate3, extracted_value)

    extracted_value = "abc"
    predicate4 = "x like 'xyz'"
    assert not PredicateExpressionParser.evaluate(predicate4, extracted_value)


def test_invalid_syntax():
    extracted_value = 5
    predicate1 = "(x > 1) xyzandabc (x < 10)"
    with pytest.raises(SyntaxError) as einfo:
        PredicateExpressionParser.evaluate(predicate1, extracted_value)
    assert "only one of AND or OR is allowed between expressions" in str(einfo.value)
