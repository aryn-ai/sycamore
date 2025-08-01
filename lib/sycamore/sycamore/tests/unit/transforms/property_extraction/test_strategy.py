from sycamore.transforms.property_extraction.strategy import TakeFirstTrimSchema, RichProperty
from sycamore.schema import NamedProperty, Schema, SchemaField
from sycamore.schema import SchemaV2, StringProperty, ObjectProperty, ArrayProperty, DataType


class TestSchemaUpdateStrategy:
    def test_takefirst_trimschema(self):
        start_schema = SchemaV2(
            properties=[
                NamedProperty(name="a", type=StringProperty()),
                NamedProperty(
                    name="b",
                    type=ObjectProperty(
                        properties=[
                            NamedProperty(name="b1", type=StringProperty()),
                            NamedProperty(name="b2", type=StringProperty()),
                        ]
                    ),
                ),
                NamedProperty(name="c", type=StringProperty()),
            ]
        )
        strat = TakeFirstTrimSchema()

        props: dict[str, RichProperty] = dict()
        p1 = {"a": RichProperty(name="a", type=DataType.STRING, value="a1")}
        sur1 = strat.update_schema(start_schema, p1, props)
        assert not sur1.completed
        assert "a" in sur1.out_fields
        assert len(sur1.out_schema.properties) == 2

        p2 = {
            "a": RichProperty(name="a", type=DataType.STRING, value="a2"),
            "b": RichProperty(
                name="b", type=DataType.OBJECT, value={"b1": RichProperty(name="b1", type=DataType.STRING, value="b22")}
            ),
        }
        sur2 = strat.update_schema(sur1.out_schema, p2, sur1.out_fields)
        print(sur2.out_fields)
        assert not sur2.completed
        assert sur2.out_fields["a"].value == "a1"
        assert sur2.out_fields["b"].value["b1"].value == "b22"
        assert "c" not in sur2.out_fields
        assert len(sur2.out_schema.properties) == 2

        p3 = {
            "c": RichProperty(name="c", type=DataType.STRING, value="c3"),
            "b": RichProperty(
                name="b", type=DataType.OBJECT, value={"b2": RichProperty(name="b2", type=DataType.STRING, value="b23")}
            ),
        }
        sur3 = strat.update_schema(sur2.out_schema, p3, sur2.out_fields)
        assert sur3.completed
        assert sur3.out_fields["a"].value == "a1"
        assert sur3.out_fields["b"].value["b1"].value == "b22"
        assert sur3.out_fields["b"].value["b2"].value == "b23"
        assert sur3.out_fields["c"].value == "c3"

    def test_takefirst_trimschema_with_array(self):
        start_schema = Schema(
            fields=[
                SchemaField(name="a", field_type="array"),
                SchemaField(name="b", field_type="string"),
                SchemaField(name="c", field_type="string"),
            ]
        )
        start_schema = SchemaV2(
            properties=[
                NamedProperty(name="a", type=ArrayProperty(item_type=StringProperty())),
                NamedProperty(name="b", type=StringProperty()),
                NamedProperty(name="c", type=StringProperty()),
            ]
        )
        strat = TakeFirstTrimSchema()

        props: dict[str, RichProperty] = dict()
        p1 = {
            "a": RichProperty(
                name="a", type=DataType.ARRAY, value=[RichProperty(name=None, type=DataType.STRING, value="a1")]
            )
        }
        sur1 = strat.update_schema(start_schema, p1, props)
        assert not sur1.completed
        assert "a" in sur1.out_fields
        assert sur1.out_fields["a"].value[0].value == "a1"
        assert len(sur1.out_fields["a"].value) == 1
        assert len(sur1.out_schema.fields) == 3

        p2 = {
            "a": RichProperty(
                name="a", type=DataType.ARRAY, value=[RichProperty(name=None, type=DataType.STRING, value="a2")]
            ),
            "b": RichProperty(name="b", type=DataType.STRING, value="b2"),
        }
        sur2 = strat.update_schema(sur1.out_schema, p2, sur1.out_fields)
        assert not sur2.completed
        assert sur2.out_fields["a"].value[0].value == "a1"
        assert sur2.out_fields["a"].value[1].value == "a2"
        assert sur2.out_fields["b"].value == "b2"
        assert "c" not in sur2.out_fields
        assert len(sur2.out_schema.fields) == 2

        p3 = {"c": RichProperty(name="c", type=DataType.STRING, value="c3")}
        sur3 = strat.update_schema(sur2.out_schema, p3, sur2.out_fields)
        assert not sur3.completed
        assert sur3.out_fields["a"].value[0].value == "a1"
        assert sur3.out_fields["a"].value[1].value == "a2"
        assert sur3.out_fields["b"].value == "b2"
        assert sur3.out_fields["c"].value == "c3"
