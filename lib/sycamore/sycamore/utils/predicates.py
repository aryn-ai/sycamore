from typing import Any

from sycamore.schema import DataType

allowed_operators = {
    DataType.STRING: ["like", "=="],
    DataType.FLOAT: [">", ">=", "<=", "<", "!=", "=="],
    DataType.INT: [">", ">=", "<=", "<", "!=", "=="],
    DataType.BOOL: ["is", "is not", "==", "!="]
}


class Expression:

    def __init__(self, property_type: DataType = DataType.STRING, extracted: str = "", value: str = "", op: str = ""):
        self.property_type = property_type
        self.extracted = extracted
        self.value = value
        self.op = op

        assert self.property_type in allowed_operators, f"Unsupported property_type: {self.property_type}"
        self._convert_value()

    def validate_op(self, op: str):
        if op not in allowed_operators[self.property_type]:
            raise SyntaxError(f"Invalid syntax: '{op}' is not allowed/supported for {self.property_type}")

    def _convert_value(self):
        match self.property_type:
            case DataType.STRING:
                self.value = str(self.value)
            case DataType.FLOAT:
                self.value = float(self.value)
            case DataType.INT:
                self.value = int(self.value)
            case DataType.BOOL:
                self.value = bool(self.value)
            case _:
                raise ValueError(f"Unsupported property_type for value conversion: {self.property_type}")

    def evaluate(self) -> bool:
        match self.property_type:
            case DataType.STRING:
                match self.op:
                    case "like":
                        return self.extracted in self.value
                    case "==":
                        return self.extracted == self.value
            case DataType.FLOAT | DataType.INT:
                match self.op:
                    case "<":
                        return self.extracted < self.value
                    case "<=":
                        return self.extracted <= self.value
                    case ">":
                        return self.extracted > self.value
                    case ">=":
                        return self.extracted >= self.value
                    case "!=":
                        return self.extracted != self.value
                    case "==":
                        return self.extracted == self.value
            case DataType.BOOL:
                match self.op:
                    case ("is", "=="):
                        return self.extracted == self.value
                    case ("is not", "!="):
                        return self.extracted != self.value
            case _:
                raise ValueError(f"Unable to evaluate expression: {self}")
        raise ValueError(f"Unable to evaluate expression: {self}")


class PredicateExpressionParser:
    """
    This parser accepts a limited set of expressions for a limited set of operations and data types.
    Valid expressions that this parser can evaluate are in the following format:
    valid_expression = expression
                       | expression AND expression
                       | expression OR expression

    expression = ( x op value )

    Valid operations depend on the data type of the property ('x').  See the Expression class for more details.
    """

    def parse_expr(self, expr: str, property_type: DataType, extracted_value) -> Expression:
        tokens = expr.split()

        if len(tokens) != 3:
            raise SyntaxError("Invalid syntax: expression must be in the form of 'x' <op> <value>, e.g. x > 1")

        if tokens[0] != "x":
            raise SyntaxError(f"Invalid syntax: the property reference must always be 'x' ('{tokens[0]}' -> 'x')")

        op = tokens[1]
        value = tokens[2]

        e = Expression(property_type=property_type, extracted=extracted_value, value=value, op=op)
        e.validate_op(op)
        return e

    def evaluate(self, expr: str, property_type: DataType, extracted_value: Any) -> bool:
        expr = expr.strip()
        if expr.startswith("("):
            if not expr.endswith(")"):
                raise SyntaxError("Invalid syntax: missing a closing parenthesis")
            idx = expr.find(")", 1)
            e = self.parse_expr(expr[1:idx], property_type, extracted_value)

            result = e.evaluate()
            if (idx2 := expr.find("(", idx + 1)) != -1:
                is_and = "and" in expr[idx + 1:idx2].lower()
                is_or = "or" in expr[idx + 1:idx2].lower()

                if not (is_and or is_or):
                    raise SyntaxError("Invalid syntax: two expressions must be joined by an AND or OR")

                if is_and and is_or:
                    raise SyntaxError("Invalid syntax: only one of AND or OR is allowed")

                idx3 = expr.find(")", idx2 + 1)
                if idx3 == -1:
                    raise SyntaxError("Invalid syntax: missing a closing parenthesis for the second expression")

                e2 = self.parse_expr(expr[idx2 + 1:idx3], property_type, extracted_value)

                if is_and:
                    result = result and e2.evaluate()
                elif is_or:
                    result = result or e2.evaluate()

                return result
            return result
        return self.parse_expr(expr, property_type, extracted_value).evaluate()

