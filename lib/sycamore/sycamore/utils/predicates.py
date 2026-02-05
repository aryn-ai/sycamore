import ast
from typing import Any, Optional

from sycamore.datatype import DataType

allowed_operators = {
    DataType.STRING: ["like", "==", "!="],
    DataType.FLOAT: [">", ">=", "<=", "<", "!=", "=="],
    DataType.INT: [">", ">=", "<=", "<", "!=", "=="],
    DataType.BOOL: ["is", "is not", "==", "!="],
}


class Expression:

    def __init__(self, extracted: Any, op: str, value: Any):
        self.extracted = extracted
        self.value = value
        self.op = op
        self.property_type: DataType = DataType.from_python(extracted)
        self._convert_value()

    def validate_op(self, op: str):
        if op not in allowed_operators[self.property_type]:
            raise SyntaxError(f"Invalid syntax: '{op}' is not allowed/supported for {self.property_type}")

    def _convert_value(self):
        match self.property_type:
            case DataType.STRING:
                self.value = ast.literal_eval(self.value)
            case DataType.FLOAT:
                self.value = float(self.value)
            case DataType.INT:
                self.value = int(self.value)
            case DataType.BOOL:
                if self.value.lower() not in ("true", "false"):
                    raise SyntaxError("Boolean data type can only be compared to 'True' or 'False'")
                self.value = True if self.value.lower() == "true" else False
            case _:
                raise ValueError(
                    f"Unsupported property_type for value conversion: {self.property_type} for extracted value {self.extracted}"
                )

    def evaluate(self) -> bool:
        match self.property_type:
            case DataType.STRING:
                match self.op:
                    case "like":
                        return self.extracted in self.value
                    case "==":
                        return self.extracted == self.value
                    case "!=":
                        return self.extracted != self.value
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
                    case "==":
                        return self.extracted == self.value
                    case "!=":
                        return self.extracted != self.value
            case _:
                raise ValueError(f"Unable to evaluate expression: {self}")
        raise ValueError(f"Unable to evaluate expression: {self}")


class PredicateExpressionParser:
    """
    This parser accepts a limited set of expressions for a limited set of operations and data types.
    Valid expressions that this parser can evaluate are in the following format:
    valid_expression = expression
                        | (expression)
                       | (expression) AND (expression)
                       | (expression) OR (expression)

    expression = x op value

    Valid operations depend on the data type of the property ('x').  See the Expression class for more details.
    """

    @staticmethod
    def parse_expr(expr: str, extracted_value: Any) -> Optional[Expression]:
        tokens = expr.split()

        if len(tokens) != 3:
            raise SyntaxError("Invalid syntax: expression must be in the form of 'x' <op> <value>, e.g. x > 1")

        if tokens[0] != "x":
            raise SyntaxError(f"Invalid syntax: the property reference must always be 'x' ('{tokens[0]}' -> 'x')")

        op = tokens[1]
        value = tokens[2]

        if extracted_value is None:  # parse only
            return None

        e = Expression(extracted=extracted_value, op=op, value=value)
        e.validate_op(op)
        return e

    @staticmethod
    def evaluate(expr: str, extracted_value: Any) -> bool:
        """

        Args:
            expr: the expression string to be evaluated to true or false
            extracted_value: if None, we perform parsing only and throw SyntaxError if expr is invalid.

        Returns: True if expression evaluates to true; False otherwise

        """
        expr = expr.strip()
        result: bool = False
        if expr.startswith("("):
            if not expr.endswith(")"):
                raise SyntaxError("Invalid syntax: missing a closing parenthesis")
            idx = expr.find(")", 1)
            e = PredicateExpressionParser.parse_expr(expr[1:idx], extracted_value)

            if e and extracted_value is not None:
                result = e.evaluate()
            if (idx2 := expr.find("(", idx + 1)) != -1:
                if expr[idx + 1 : idx2].lower().strip() not in ["and", "or"]:
                    raise SyntaxError("Invalid syntax: only one of AND or OR is allowed between expressions")
                is_and = "and" in expr[idx + 1 : idx2].lower()
                is_or = "or" in expr[idx + 1 : idx2].lower()

                if not (is_and or is_or):
                    raise SyntaxError("Invalid syntax: two expressions must be joined by an AND or OR")

                if is_and and is_or:
                    raise SyntaxError("Invalid syntax: only one of AND or OR is allowed")

                idx3 = expr.find(")", idx2 + 1)
                if idx3 == -1:
                    raise SyntaxError("Invalid syntax: missing a closing parenthesis for the second expression")

                e2 = PredicateExpressionParser.parse_expr(expr[idx2 + 1 : idx3], extracted_value)

                if e2 and extracted_value is not None:
                    if is_and:
                        result = result and e2.evaluate()
                    elif is_or:
                        result = result or e2.evaluate()

                return result
            return result

        e = PredicateExpressionParser.parse_expr(expr, extracted_value)
        if e and extracted_value is not None:
            return e.evaluate()
        return result
