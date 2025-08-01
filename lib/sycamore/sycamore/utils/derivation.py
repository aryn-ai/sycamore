from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class UnitConverter:
    def __init__(self):
        # Graph representation: adjacency list with conversion rates
        self.graph = defaultdict(dict)
        self.add_conversion("SF", "SM", 0.092903)
        self.add_conversion("acre", "SF", 43560.0)
        self.add_conversion("hectare", "SM", 10000.0)
        self.add_conversion("acre", "hectare", 0.404686)

    def add_conversion(self, from_unit: str, to_unit: str, factor: float) -> None:
        self.graph[from_unit][to_unit] = factor
        self.graph[to_unit][from_unit] = 1 / factor  # Inverse

    def find_conversion_rate(self, from_unit: str, to_unit: str) -> float:
        # BFS to find shortest path and cumulative conversion factor
        if from_unit == to_unit:
            return 1.0
        visited = set()
        queue = deque([(from_unit, 1.0)])  # (current_unit, cumulative_factor)
        while queue:
            current_unit, current_factor = queue.popleft()
            visited.add(current_unit)
            for neighbor, factor in self.graph[current_unit].items():
                if neighbor == to_unit:
                    return current_factor * factor
                if neighbor not in visited:
                    queue.append((neighbor, current_factor * factor))
        raise ValueError(f"No conversion path found between {from_unit} and {to_unit}")

    def _calculate_rounding_factor(self, value: float, additional_digits: int = 1) -> float:
        """Calculate rounding factor based on input value precision."""
        s = str(value)  # Use Python's choice for precision
        if "." in s:  # digits beyond decimal point
            factor = 1
            for i in range(1, len(s) + 1):
                if s[-i] == '.':
                    break
                factor /= 10
        else:  # no digits beyond decimal point
            factor = 1
            for i in range(1, len(s) + 1):
                if s[-i] != '0':
                    break
                factor *= 10
        factor /= 10**additional_digits
        return factor

    def convert_exact(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value without any rounding."""
        rate = self.find_conversion_rate(from_unit, to_unit)
        return value * rate

    def convert(self, value: float, from_unit: str, to_unit: str, additional_digits: int = 1) -> float:
        """Convert value with precision-aware rounding."""
        rate = self.find_conversion_rate(from_unit, to_unit)
        exact_value = value * rate
        factor = self._calculate_rounding_factor(value, additional_digits)
        rounded_value = factor * round(exact_value / factor, 0)
        return rounded_value

    def __call__(self, value: float, from_unit: str, to_unit: str) -> float:
        """Make the unit converter callable for easy conversion."""
        return self.convert(value, from_unit, to_unit)

    def check(self, from_unit: str, to_unit: str, value1: float, value2: float) -> bool:
        converted_value = self.convert_exact(value1, from_unit, to_unit)
        return value2 and abs(converted_value - value2) / value2 < 0.05


class PropertyDerivation:
    def __init__(self, properties: dict[str, float], metadata: dict[str, list[str]] = {}, property_to_unit: dict[str, str] = {}, unit_converter: UnitConverter = None) -> None:
        self.properties = properties
        self.metadata = metadata
        self.unit_map = property_to_unit  # Rename for clarity
        self.unit_converter = unit_converter or UnitConverter()
        self.group = []

    def add_conversion(self, from_unit: str, to_unit: str, factor: float) -> None:
        """Delegate to the unit converter."""
        self.unit_converter.add_conversion(from_unit, to_unit, factor)

    def convert(self, value: float, from_unit: str, to_unit: str, additional_digits: int = 1) -> float:
        """Delegate to the unit converter."""
        return self.unit_converter.convert(value, from_unit, to_unit, additional_digits)

    def convert_exact(self, value: float, from_unit: str, to_unit: str) -> float:
        """Delegate to the unit converter."""
        return self.unit_converter.convert_exact(value, from_unit, to_unit)

    def derive_conversion(self, to_property_name: str, from_property_name: str) -> bool:
        if to_property_name in self.properties:
            return False  # Don't overwrite existing values

        from_value = self.properties.get(from_property_name)
        if from_value is None:
            return False

        if to_property_name in self.unit_map and from_property_name in self.unit_map:
            to_unit = self.unit_map[to_property_name]
            from_unit = self.unit_map[from_property_name]

            try:
                converted_value = self.unit_converter.convert(from_value, from_unit, to_unit)
                self.properties[to_property_name] = converted_value
                self.metadata[to_property_name] = [from_property_name]
                return True
            except Exception:
                return False
        return False

    def _derive_all_conversion(self, to_property_name: str, from_property_name: str) -> bool:
        if to_property_name not in self.properties and from_property_name not in self.properties:
            return False
        value1 = self.properties.get(from_property_name)
        if value1 is None:
            return False
        if to_property_name in self.unit_map and from_property_name in self.unit_map:
            to_unit = self.unit_map[to_property_name]
            from_unit = self.unit_map[from_property_name]
            if value1 is not None:
                converted_value = self.unit_converter.convert(value1, from_unit, to_unit)
                self.properties[to_property_name] = converted_value
                self.metadata[to_property_name] = [from_property_name]
                return True
        return False

    def fill_missing_units(self, list_of_fields: list[str]) -> None:
        fields_with_value = [field for field in list_of_fields if self.properties.get(field) is not None]
        if len(fields_with_value) == 0:
            return
        reference_field = fields_with_value[0]
        for field in list_of_fields:
            if field == reference_field:
                continue

            if self.properties.get(field) is None:
                self._derive_all_conversion(field, reference_field)

    def unit_group(self, group: list[str]) -> None:
        ref_property = None
        for g in group:
            assert g in self.unit_map, f"Have a property {g} without units requested for conversion"
        for g in group:
             if self.properties.get(g) is not None:
                  ref_property = g
                  break
        ref_value, ref_unit = self.properties[ref_property], self.unit_map[ref_property]
        for g in group:
            if self.properties.get(g) is not None:
                continue
              # make __call__ on unit_converter do convert()
            converted_value = self.unit_converter.convert_exact(ref_value, ref_unit, self.unit_map[g])
            self.properties[g] = converted_value
            self.metadata[g] = [ref_property]  # Set metadata to track the source
            if self.properties[g] is None:
                logger.warning(f"Unable to convert {ref_unit} to {self.unit_map[g]} within a unit group for properties {ref_property} to {g}")
        # here's where I'd check consistency, but I don't see that happening in the existing code.
        self.group.append(group)

    def fill_from_formula(self, formula_string: str) -> dict[str, float] | None:
        ops = "+-*/"
        left, right = [part.strip() for part in formula_string.split("=")]
        if any(op in left for op in ops):
            expr, target = left, right  #  a op b = target
        else:
            expr, target = right, left  #  target = a op b
        for op in ops:
            if op in expr:
                a_sym, b_sym = [t.strip() for t in expr.split(op, 1)]
                operator_symbol = op
                break
        else:
            raise ValueError("No arithmetic operator found in formula")

        def value_of(token: str) -> float | None:
            return self.properties.get(token)

        val_a = value_of(a_sym)
        val_b = value_of(b_sym)
        val_target = value_of(target)

        known = sum(x is not None for x in (val_a, val_b, val_target))
        if known != 2:
            return None

        #  compute the missing one
        if val_target is None:  # find target
            if operator_symbol == "+":
                val_target = val_a + val_b
            elif operator_symbol == "-":
                val_target = val_a - val_b
            elif operator_symbol == "*":
                val_target = val_a * val_b
            elif operator_symbol == "/":
                val_target = val_a / val_b
            self.properties[target] = val_target

        elif val_a is None:  # find a
            if operator_symbol == "+":
                val_a = val_target - val_b
            elif operator_symbol == "-":  # target = a - b
                val_a = val_target + val_b
            elif operator_symbol == "*":
                val_a = val_target / val_b
            elif operator_symbol == "/":  # target = a / b
                val_a = val_target * val_b
            self.properties[a_sym] = val_a

        elif val_b is None:  # find b
            if operator_symbol == "+":
                val_b = val_target - val_a
            elif operator_symbol == "-":  # target = a - b
                val_b = val_a - val_target
            elif operator_symbol == "*":
                val_b = val_target / val_a
            elif operator_symbol == "/":  # target = a / b
                val_b = val_a / val_target
            self.properties[b_sym] = val_b

        return self.properties


def main():
    properties = {"airdistance": 101, "altitude_cm": 12, "airPerAl": 20}
    property_to_unit = {
        "airdistance": "cm",
        "airdistance_m": "m",
        "altitude": "mm",
        "altitude_cm": "cm",
        "altitude_m": "m",
    }

    ud = PropertyDerivation(properties, property_to_unit=property_to_unit)
    ud.add_conversion("m", "cm", 100.0)
    ud.add_conversion("cm", "mm", 10.0)
    ud.add_conversion("m", "ft", 3.28084)  # Add meter to feet conversion

    ud.unit_group(["airdistance", "airdistance_m"])
    ud.unit_group(["altitude", "altitude_cm"])

    ud.fill_from_formula("airPerAl = airspeed_m / altitude_m")

    print(ud.properties, ud.metadata)
    
    # Test the new precision-aware conversion
    print("\nTesting precision-aware conversion:")
    print(f"1000m -> {ud.convert(1000, 'm', 'ft')}ft (with precision-aware rounding)")
    print(f"1000m -> {ud.convert_exact(1000, 'm', 'ft')}ft (exact conversion)")
    print(f"100.5m -> {ud.convert(100.5, 'm', 'ft')}ft (with precision-aware rounding)")
    print(f"100.5m -> {ud.convert_exact(100.5, 'm', 'ft')}ft (exact conversion)")


if __name__ == "__main__":
    main()
