from collections import deque, defaultdict


class UnitConverter:
    def __init__(self):
        # Graph representation: adjacency list with conversion rates
        self.graph = defaultdict(dict)
        self.add_conversion("SF", "SM", 0.092903)
        self.add_conversion("acre", "SF", 43560.0)
        self.add_conversion("hectare", "SM", 10000.0)
        self.add_conversion("acre", "hectare", 0.404686)

    def add_conversion(self, from_unit, to_unit, factor):
        self.graph[from_unit][to_unit] = factor
        self.graph[to_unit][from_unit] = 1 / factor  # Inverse

    def find_conversion_rate(self, from_unit, to_unit):
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

    def convert(self, value, from_unit, to_unit):
        rate = self.find_conversion_rate(from_unit, to_unit)
        return round(value * rate, 6)

    def check(self, from_unit, to_unit, value1, value2):
        converted_value = self.convert(value1, from_unit, to_unit)
        return value2 and abs(converted_value - value2) / value2 < 0.05


class UnitDerivation:
    def __init__(self, properties, metadata, unitConverter: UnitConverter, unit_map=None):
        self.properties = properties
        self.metadata = metadata
        self.unitConverter = unitConverter
        self.unit_map = unit_map or {}

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
                converted_value = self.unitConverter.convert(from_value, from_unit, to_unit)
                self.properties[to_property_name] = converted_value
                self.metadata[to_property_name] = [from_property_name]
                return True
            except Exception:
                return False
        return False

    def derive_all_conversion(self, to_property_name, from_property_name, unit_map: None):
        self.unit_map = unit_map or self.unit_map
        if to_property_name not in self.properties and from_property_name not in self.properties:
            return False
        value1 = self.properties.get(from_property_name)
        if value1 is None:
            return False
        if to_property_name in self.unit_map and from_property_name in self.unit_map:
            to_unit = self.unit_map[to_property_name]
            from_unit = self.unit_map[from_property_name]
            if value1 is not None:
                converted_value = self.unitConverter.convert(value1, from_unit, to_unit)
                self.properties[to_property_name] = converted_value
                self.metadata[to_property_name] = [from_property_name]
                return True
        return False

    def fill_missing_units(self, list_of_fields, unit_map):
        fields_with_value = [field for field in list_of_fields if self.properties.get(field) is not None]
        if len(fields_with_value) == 0:
            return
        reference_field = fields_with_value[0]
        for field in list_of_fields:
            if field == reference_field:
                continue

            if self.properties.get(field) is None:
                self.derive_all_conversion(field, reference_field, unit_map)


class PropertyDerivation(UnitConverter, UnitDerivation):
    def __init__(self, properties, metadata=None):

        self.properties = properties
        self.metadata = metadata or {}
        UnitConverter.__init__(self)
        self.unit_map = {}
        UnitDerivation.__init__(self, self.properties, self.metadata, self, self.unit_map)
        self.group = []

    def unit_group(self, group):
        temp_group = []
        for property, unit in group:
            self.unit_map[property] = unit
            temp_group.append(property)
        self.fill_missing_units(temp_group, self.unit_map)
        self.group.append(temp_group)

    def fill_from_formula(self, formula_string: str):
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

        def value_of(token):
            return self.properties.get(token)

        val_a = value_of(a_sym)
        val_b = value_of(b_sym)
        val_target = value_of(target)

        known = sum(x is not None for x in (val_a, val_b, val_target))
        if known != 2:
            return

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


properties = {"airspeed": 101, "altitude_cm": 12, "airPerAl": 20}
ud = PropertyDerivation(properties)
ud.add_conversion("m", "cm", 100.0)
ud.add_conversion("cm", "mm", 10.0)

ud.unit_group([("airspeed", "cm"), ("airspeed_m", "m")])
ud.unit_group([("altitude", "mm"), ("altitude_cm", "cm")])

ud.fill_from_formula("airPerAl = airspeed_m / altitude_m")

print(properties)
