from sycamore.utils.predicates import PredicateExpressionParser


def test_parse():
    parser = PredicateExpressionParser()

    extracted_value = 2

    predicate1 = "x > 1"
    assert parser.evaluate(predicate1, extracted_value)

    predicate2 = "x > 2"
    assert not parser.evaluate(predicate2, extracted_value)

    predicate3 = "(x == 2)"
    assert parser.evaluate(predicate3, extracted_value)

    predicate4 = "(x >= 1) AND (x < 10)"
    assert parser.evaluate(predicate4, extracted_value)
