import sys
import os
from typing import Dict, Any
import re
from collections import deque, defaultdict

root_dir = os.path.normpath(os.path.dirname(__file__) + "/../..")
sys.path.append(root_dir)


field_type_lookup: Dict[str, Any] = {}

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
        self.unit_map = unit_map or  {
        }

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

    def derive_ratio(self, ratio_name: str, numerator_prop: str, denominator_prop: str) -> bool:
        if ratio_name in self.properties:
            return False  # Don't overwrite existing values

        numerator = self.properties.get(numerator_prop)
        denominator = self.properties.get(denominator_prop)

        if numerator is None or denominator is None or denominator == 0:
            return False

        try:
            ratio_value = numerator / denominator
            self.properties[ratio_name] = round(ratio_value, 4)
            self.metadata[ratio_name] = [numerator_prop, denominator_prop]
            return True
        except (ZeroDivisionError, TypeError):
            return False

    def derive_percentage(
        self, percentage_name: str, part_prop: str, total_prop: str, max_value: float = 100.0
    ) -> bool:
        if percentage_name in self.properties:
            return False

        part = self.properties.get(part_prop)
        total = self.properties.get(total_prop)

        if part is None or total is None or total == 0:
            return False

        try:
            percentage = min((part / total) * 100.0, max_value)
            self.properties[percentage_name] = round(percentage, 2)
            self.metadata[percentage_name] = [part_prop, total_prop]
            return True
        except (ZeroDivisionError, TypeError):
            return False

    def derive_per_unit_metric(
        self, metric_name: str, total_prop: str, unit_prop: str, unit_divisor: float = 1.0, precision: int = 2
    ) -> bool:
        if metric_name in self.properties:
            return False
        total = self.properties.get(total_prop)
        units = self.properties.get(unit_prop)

        if total is None or units is None or units == 0:
            return False

        try:
            per_unit_value = total / (units / unit_divisor)
            self.properties[metric_name] = round(per_unit_value, precision)
            self.metadata[metric_name] = [total_prop, unit_prop]
            return True
        except (ZeroDivisionError, TypeError):
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
                if isinstance(value1, str):
                    value1 = extract_numbers(value1)
                converted_value = self.unitConverter.convert(value1, from_unit, to_unit)
                self.properties[to_property_name] = converted_value
                self.metadata[to_property_name] = [from_property_name]
                return True
        return False

    def fill_missing_units(self, list_of_fields, unit_map ):
        # Check if all fields are None
        fields_with_value = [field for field in list_of_fields if self.properties.get(field) is not None]
        if len(fields_with_value) == 0:
            return
        reference_field = fields_with_value[0]
        # Fill in missing fields using the reference field
        for field in list_of_fields:
            if field == reference_field:  # Skip the reference field itself
                continue

            if self.properties.get(field) is None:  # Only fill in if the field is None
                self.derive_all_conversion(field, reference_field, unit_map )


class PropertyDerivation(UnitConverter, UnitDerivation):
    def __init__(self, properties , metadata = None):

        self.properties = properties 
        self.metadata = metadata or {}
        UnitConverter.__init__(self)  # This calls UnitConver
        self.unit_map = {}
        UnitDerivation.__init__(self, self.properties, self.metadata, self, self.unit_map)
        self.group = []   
    
    def unit_group(self, group):
        temp_group = []
        for property,unit in group:
            self.unit_map[property] = unit
            temp_group.append(property)
        self.fill_missing_units(temp_group, self.unit_map)
        self.group.append(temp_group)

    def evaluate_formula(self, formula_string):
        left_side, target_property = formula_string.split("=")
        target_property = target_property.strip()

        # tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_/()]*|[+\-*/()]', left_side.strip())
        # expression_parts = []
        # for token in tokens:
        #     if token in ['+', '-', '*', '/']:
        #         expression_parts.append(token)
        #     else:
        #         value = 

        elements_formula = formula_string.split(" ")[::-1]
        target = elements_formula[-1]
        elements = elements_formula[:-2]
        queue = []
        for element in elements:
            if element in ['+', '-', '*', '/']:
                queue.append(element)
            else:
                temp = self.properties.get(element, None)
                if temp:
                    queue.append(str(temp))
        self.properties[target]  = eval(''.join(queue)) 
        return self.properties

properties = {'airspeed': 101, 'altitude_cm':12}
ud = PropertyDerivation(properties)
ud.add_conversion("m", "cm", 100.0)

ud.unit_group([("airspeed", "cm"), ("airspeed_m", "m")])
ud.unit_group([("altitude", "m"), ("altitude_cm", "cm")])

ud.evaluate_formula("airPerAl = airspeed_m / altitude")

print(properties)