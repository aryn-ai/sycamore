from sycamore.transforms.property_extraction.strategy import TakeFirstTrimSchema, RichProperty
from sycamore.schema import Schema, SchemaField


class TestSchemaUpdateStrategy:
    def test_takefirst_trimschema(self):
        start_schema = Schema(
            fields=[
                SchemaField(name="a", field_type="string"),
                SchemaField(name="b", field_type="string"),
                SchemaField(name="c", field_type="string"),
            ]
        )
        strat = TakeFirstTrimSchema()

        props: dict[str, RichProperty] = dict()
        p1 = {"a": RichProperty(name="a", type="string", value="a1")}
        sur1 = strat.update_schema(start_schema, p1, props)
        assert not sur1.completed
        assert "a" in sur1.out_fields
        assert len(sur1.out_schema.fields) == 2

        p2 = {
            "a": RichProperty(name="a", type="string", value="a2"),
            "b": RichProperty(name="b", type="string", value="b2"),
        }
        sur2 = strat.update_schema(sur1.out_schema, p2, sur1.out_fields)
        assert not sur2.completed
        assert sur2.out_fields["a"].value == "a1"
        assert sur2.out_fields["b"].value == "b2"
        assert "c" not in sur2.out_fields
        assert len(sur2.out_schema.fields) == 1

        p3 = {"c": RichProperty(name="c", type="string", value="c3")}
        sur3 = strat.update_schema(sur2.out_schema, p3, sur2.out_fields)
        assert sur3.completed
        assert sur2.out_fields["a"].value == "a1"
        assert sur2.out_fields["b"].value == "b2"
        assert sur2.out_fields["c"].value == "c3"

    def test_takefirst_trimschema_with_array(self):
        start_schema = Schema(
            fields=[
                SchemaField(name="a", field_type="array"),
                SchemaField(name="b", field_type="string"),
                SchemaField(name="c", field_type="string"),
            ]
        )
        strat = TakeFirstTrimSchema()

        props: dict[str, RichProperty] = dict()
        p1 = {"a": RichProperty(name="a", type="string", value=["a1"])}
        sur1 = strat.update_schema(start_schema, p1, props)
        assert not sur1.completed
        assert "a" in sur1.out_fields
        assert len(sur1.out_schema.fields) == 3

        p2 = {
            "a": RichProperty(name="a", type="string", value=["a2"]),
            "b": RichProperty(name="b", type="string", value="b2"),
        }
        sur2 = strat.update_schema(sur1.out_schema, p2, sur1.out_fields)
        assert not sur2.completed
        assert sur2.out_fields["a"].value == ["a1", "a2"]
        assert sur2.out_fields["b"].value == "b2"
        assert "c" not in sur2.out_fields
        assert len(sur2.out_schema.fields) == 2

        p3 = {"c": RichProperty(name="c", type="string", value="c3")}
        sur3 = strat.update_schema(sur2.out_schema, p3, sur2.out_fields)
        assert not sur3.completed
        assert sur2.out_fields["a"].value == ["a1", "a2"]
        assert sur2.out_fields["b"].value == "b2"
        assert sur2.out_fields["c"].value == "c3"
