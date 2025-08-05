import logging

from unittest.mock import MagicMock

from sycamore.query.schema import OpenSearchSchemaFetcher
from sycamore.schema import make_named_property

logging.getLogger("sycamore.query.schema").setLevel(logging.DEBUG)


def test_opensearch_schema():
    mock_client = MagicMock()
    mock_client.get_field_mapping.return_value = {
        "test_index": {
            "mappings": {
                "properties.entity.day": {"full_name": "properties.entity.day", "mapping": {"day": {"type": "date"}}},
                "properties.entity.aircraft": {
                    "full_name": "properties.entity.aircraft",
                    "mapping": {
                        "aircraft": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
                "properties.entity.location": {
                    "full_name": "properties.entity.location",
                    "mapping": {
                        "location": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
                "properties.entity.weather": {
                    "full_name": "properties.entity.weather",
                    "mapping": {
                        "weather": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
                "properties.entity.test_prop": {
                    "full_name": "properties.entity.test_prop",
                    "mapping": {
                        "weather": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
                "properties.entity.colors": {
                    "full_name": "properties.entity.colors",
                    "mapping": {
                        "colors": {"type": "array", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
                "properties.entity.flavors": {
                    "full_name": "properties.entity.flavors",
                    "mapping": {
                        "flavors": {"type": "array", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
                "properties.entity.airspeed": {
                    "full_name": "properties.entity.airspeed",
                    "mapping": {
                        "airspeed": {"type": "array", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
                "properties.entity.count": {
                    "full_name": "properties.entity.count",
                    "mapping": {
                        "count": {"type": "array", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
                "properties.entity.weird": {
                    "full_name": "properties.entity.weird",
                    "mapping": {
                        "weird": {"type": "array", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
                "properties.happiness": {
                    "full_name": "properties.happiness",
                    "mapping": {
                        "happiness": {"type": "array", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
                    },
                },
            }
        }
    }

    mock_query_executor = MagicMock()

    mock_random_sample = {
        "result": {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "properties": {
                                "entity": {
                                    "day": "2021-01-01",
                                    "aircraft": "Boeing 747",
                                    "colors": ["red", "blue"],
                                    "flavors": "vanilla",
                                    "airspeed": 42,
                                    "count": 3,
                                    "weird": True,
                                },
                                "happiness": "yes",
                            }
                        }
                    },
                    {
                        "_source": {
                            "properties": {
                                "entity": {
                                    "day": "2021-01-02",
                                    "aircraft": "Airbus A380",
                                    "weather": "Sunny",
                                    "colors": [],
                                    "flavors": ["chocolate", "strawberry"],
                                    "airspeed": 41.5,
                                    "count": 0,
                                    "weird": 500,
                                }
                            }
                        }
                    },
                    {
                        "_source": {
                            "properties": {
                                "entity": {
                                    "day": "2021-01-02",
                                    "aircraft": "Airbus A380",
                                    "weather": "Sunny",
                                    "colors": "yellow",
                                    "flavors": "banana",
                                    "airspeed": 48,
                                    "count": 67,
                                    "weird": "alphabetical",
                                }
                            }
                        }
                    },
                    {
                        "_source": {
                            "properties": {
                                "entity": {
                                    "day": "2021-01-02",
                                    "aircraft": "Flight b741",
                                    "airspeed": 74.1,
                                }
                            }
                        }
                    },
                ]
            }
        }
    }

    # this is asserting we only take OpenSearchSchemaFetcher.NUM_EXAMPLE_VALUES examples
    for i in range(0, OpenSearchSchemaFetcher.NUM_EXAMPLE_VALUES + 5):
        mock_random_sample["result"]["hits"]["hits"] += [{"_source": {"properties": {"entity": {"test_prop": str(i)}}}}]
    # Note that there are no values for 'location' here.
    mock_query_executor.query.return_value = mock_random_sample

    fetcher = OpenSearchSchemaFetcher(mock_client, "test_index", mock_query_executor)
    got = fetcher.get_schema()
    properties_map = {p.name: p for p in got.properties}

    assert properties_map["text_representation"] == make_named_property(
        name="text_representation",
        type="string",
        description="A text representation of the document. Can be assumed to have all other details.",
    )

    assert properties_map["properties.entity.day"].type.type == "string"
    assert set(properties_map["properties.entity.day"].type.examples) == {"2021-01-01", "2021-01-02"}
    assert properties_map["properties.entity.aircraft"].type.type == "string"
    assert set(properties_map["properties.entity.aircraft"].type.examples) == {
        "Boeing 747",
        "Airbus A380",
        "Flight b741",
    }
    assert properties_map["properties.entity.weather"] == make_named_property(
        name="properties.entity.weather", type="string", examples=["Sunny"]
    )

    # A mix of lists and singletons gets promoted to a list.

    assert properties_map["properties.entity.colors"].type.type == "array"
    assert set(properties_map["properties.entity.colors"].type.examples) == {"['red', 'blue']", "[]", "['yellow']"}

    assert properties_map["properties.entity.test_prop"].type.type == "string"
    assert set(properties_map["properties.entity.test_prop"].type.examples) == {
        str(i) for i in range(OpenSearchSchemaFetcher.NUM_EXAMPLE_VALUES)
    }

    # Ints get promoted to floats when there is a mix of sample values.
    assert properties_map["properties.entity.airspeed"].type.type == "float"
    assert set(properties_map["properties.entity.airspeed"].type.examples) == {"41.5", "42", "48", "74.1"}

    # Ints stay ints when there is no mix.
    assert properties_map["properties.entity.count"].type.type == "int"
    assert set(properties_map["properties.entity.count"].type.examples) == {"3", "0", "67"}

    # Mixed type gets promoted to string.
    assert properties_map["properties.entity.weird"].type.type == "string"
    assert set(properties_map["properties.entity.weird"].type.examples) == {"500", "True", "alphabetical"}

    # Check that fields with a single sample are retained.
    assert properties_map["properties.happiness"] == make_named_property(
        name="properties.happiness", type="string", examples=["yes"]
    )

    # Check that fields with no samples are ignored.
    assert "properties.entity.location" not in properties_map
