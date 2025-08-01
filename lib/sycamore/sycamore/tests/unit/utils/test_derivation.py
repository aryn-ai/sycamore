import unittest

# Adapt this import to the real location of the production code
from sycamore.utils.derivation import UnitConverter, PropertyDerivation


class TestUnitConverter(unittest.TestCase):
    def setUp(self):
        self.conv = UnitConverter()

    # ------------------------------------------------------------------ basic routing
    def test_direct_conversion(self):
        """100 Square-Feet → Square-Metre (direct edge)"""
        result = self.conv.convert_exact(100, "SF", "SM")
        self.assertAlmostEqual(result, 9.2903, places=2)

    def test_inverse_conversion(self):
        """inverse edge (SM → SF)"""
        result = self.conv.convert_exact(9.2903, "SM", "SF")
        self.assertAlmostEqual(result, 100.0, places=0)

    def test_multi_hop_conversion(self):
        """acre → SF → SM   (BFS multi-hop)"""
        result = self.conv.convert_exact(1, "acre", "SM")
        self.assertAlmostEqual(result, 4046.856, places=0)

    def test_same_unit_shortcut(self):
        self.assertEqual(self.conv.convert(123.456, "SF", "SF"), 123.456)

    def test_missing_route_returns_none(self):
        result = self.conv.convert(1, "SF", "kg")
        self.assertIsNone(result)

    def test_callable_converter(self):
        """Test that the converter is callable"""
        result = self.conv.convert_exact(100, "SF", "SM")
        self.assertAlmostEqual(result, 9.2903, places=2)

    # ------------------------------------------------------------------ helper :check
    def test_check_method(self):
        """Value within ±5 % is accepted"""
        ok = self.conv.check("SF", "SM", 100, 9.5)  # ~2.2 % error
        nok = self.conv.check("SF", "SM", 100, 8.0)  # ~14 % error
        self.assertTrue(ok)
        self.assertFalse(nok)


class TestPropertyDerivation(unittest.TestCase):
    def setUp(self):
        # minimal example property-set
        self.properties = {"dist_cm": 250, "height_m": 2}
        self.unit_map = {
            "dist_cm": "cm",
            "dist_m": "m",
            "height_m": "m",
            "height_cm": "cm",
            "a": None,
            "b": None,
            "c": None,
        }
        self.deriver = PropertyDerivation(self.properties, property_to_unit=self.unit_map)
        self.deriver.add_conversion("m", "cm", 100)

    # --------------------------------------------------------------- conversion groups
    def test_unit_group_fills_missing_value(self):
        self.deriver.unit_group(["dist_cm", "dist_m"])  # only dist_cm present
        self.assertAlmostEqual(self.properties["dist_m"], 2.5)  # 250 cm → 2.5 m
        self.assertEqual(self.deriver.metadata["dist_m"], ["dist_cm"])

    def test_unit_group_asserts_missing_units(self):
        """Test that unit_group raises AssertionError for properties without units"""
        with self.assertRaises(AssertionError):
            self.deriver.unit_group(["dist_cm", "unknown_property"])

    def test_unit_group_skips_existing_values(self):
        """Test that unit_group doesn't overwrite existing values"""
        self.properties["dist_m"] = 999  # existing value
        self.deriver.unit_group(["dist_cm", "dist_m"])
        self.assertEqual(self.properties["dist_m"], 999)  # should remain unchanged

    # ---------------------------------------------------------------- derive_conversion
    def test_explicit_derive_conversion(self):
        filled = self.deriver.derive_conversion("height_cm", "height_m")
        self.assertTrue(filled)
        self.assertAlmostEqual(self.properties["height_cm"], 200)

    def test_derive_conversion_does_not_overwrite(self):
        self.properties["height_cm"] = 123  # already has a value
        filled = self.deriver.derive_conversion("height_cm", "height_m")
        self.assertFalse(filled)
        self.assertEqual(self.properties["height_cm"], 123)  # untouched

    # ---------------------------------------------------------------- formula handling
    def test_fill_from_formula_when_target_missing_multiply(self):
        self.properties.update({"a": 4, "b": 5, "c": None})
        self.deriver.fill_from_formula("c = a * b")
        self.assertEqual(self.properties["c"], 20)

    def test_fill_from_formula_when_target_missing_add(self):
        self.properties.update({"a": 4, "b": 5, "c": None})
        self.deriver.fill_from_formula("c = a + b")
        self.assertEqual(self.properties["c"], 9)

    def test_fill_from_formula_when_target_missing_subtract(self):
        self.properties.update({"a": 4, "b": 5, "c": None})
        self.deriver.fill_from_formula("c = a - b")
        self.assertEqual(self.properties["c"], -1)

    def test_fill_from_formula_when_target_missing_divide(self):
        self.properties.update({"a": 4, "b": 1, "c": None})
        self.deriver.fill_from_formula("c = a / b")
        self.assertEqual(self.properties["c"], 4)

    def test_fill_from_formula_when_operand_missing(self):
        self.properties.update({"a": None, "b": 5, "c": 20})
        self.deriver.fill_from_formula("c = a * b")
        self.assertEqual(self.properties["a"], 4)

    def test_formula_ignored_if_not_exactly_two_known(self):
        self.properties.update({"a": None, "b": None, "c": 20})
        # No change expected because only one value is known.
        before = dict(self.properties)
        self.deriver.fill_from_formula("c = a * b")
        self.assertDictEqual(self.properties, before)
