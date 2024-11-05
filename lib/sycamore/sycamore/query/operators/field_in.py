from sycamore import DocSet
from sycamore.query.logical_plan import Node


class FieldIn(Node):
    """Joins two databases based on a particular field.

    Values of *field_one* from
    database 1 are used to filter records of database 2 based on values of *field_two*
    in database 2. For example, consider that database 1 is {"properties.key":
    ['Cruise Ship', 'Sailboat'], "properties.count": [3, 2]} and database 2 is
    {"properties.entity.shipType": ['Jet ski', 'Canoe', 'Submarine', 'Cruise Ship'],
    "properties.entity.country": ['Australia', 'Japan', 'United States', 'Mexico'],
    "properties.entity.city": ['Sydney', 'Kyoto', 'San Francisco', 'Cabo']}. A join
    operation with *inputs* containing ids of operations that return database 1 and
    database 2, respectively, *field_one* being "properties.key", and *field_two* being
    "properties.entity.shipType", would return the database {"properties.entity.shipType":
    ['Cruise Ship'], "properties.entity.country": ['Mexico'], "properties.entity.city":
    ['Cabo']}.

    Returns a database with fields identical to those in database 2.
    """

    field_one: str
    """The field name in the first database to join on."""

    field_two: str
    """The field name in the second database to join on."""

    @property
    def input_types(self) -> set[type]:
        return {DocSet}

    @property
    def output_type(self) -> type:
        return DocSet
