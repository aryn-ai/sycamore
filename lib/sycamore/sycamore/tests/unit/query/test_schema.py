import logging

from unittest.mock import MagicMock

from sycamore.query.schema import OpenSearchSchemaFetcher, OpenSearchSchemaField

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
                "properties.entity.airspeed": {
                    "full_name": "properties.entity.airspeed",
                    "mapping": {
                        "airspeed": {"type": "array", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
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
                                "entity": {"day": "2021-01-01", "aircraft": "Boeing 747", "colors": ["red", "blue"]},
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
                                }
                            }
                        }
                    },
                ]
            }
        }
    }

    # Verify we can handle schemas where int and float are both used
    airspeeds = [500, 37.5, 300, 217.11]
    for i in airspeeds:
        mock_random_sample["result"]["hits"]["hits"].append({"_source": {"properties": {"entity": {"airspeed": i}}}})

    # Verify we tolerate schemas where the types are incompatible
    weird = [True, 500, "alphabetical"]
    for i in weird:
        mock_random_sample["result"]["hits"]["hits"].append({"_source": {"properties": {"entity": {"weird": i}}}})

    # this is asserting we only take OpenSearchSchemaFetcher.NUM_EXAMPLE_VALUES examples
    for i in range(0, OpenSearchSchemaFetcher.NUM_EXAMPLE_VALUES + 5):
        mock_random_sample["result"]["hits"]["hits"] += [{"_source": {"properties": {"entity": {"test_prop": str(i)}}}}]
    # Note that there are no values for 'location' here.
    mock_query_executor.query.return_value = mock_random_sample

    fetcher = OpenSearchSchemaFetcher(mock_client, "test_index", mock_query_executor)
    got = fetcher.get_schema()
    assert "text_representation" in got
    assert got["text_representation"] == OpenSearchSchemaField(
        type="str", samples={"Can be assumed to have all other details"}
    )
    assert "properties.entity.day" in got
    assert got["properties.entity.day"] == OpenSearchSchemaField(type="str", samples={"2021-01-01", "2021-01-02"})
    assert "properties.entity.aircraft" in got
    assert got["properties.entity.aircraft"] == OpenSearchSchemaField(type="str", samples={"Boeing 747", "Airbus A380"})
    assert "properties.entity.weather" in got
    assert got["properties.entity.weather"] == OpenSearchSchemaField(type="str", samples={"Sunny"})
    assert "properties.entity.colors" in got
    assert got["properties.entity.colors"] == OpenSearchSchemaField(
        type="list", samples={str(["red", "blue"]), str([])}
    )
    assert "properties.entity.test_prop" in got
    assert got["properties.entity.test_prop"] == OpenSearchSchemaField(
        type="str",
        samples=set([str(i) for i in range(OpenSearchSchemaFetcher.NUM_EXAMPLE_VALUES)]),
    )
    assert got["properties.entity.airspeed"] == OpenSearchSchemaField(
        type="float", samples=set([str(a) for a in airspeeds])
    )
    assert got["properties.entity.weird"] == OpenSearchSchemaField(type="bool", samples=set([str(w) for w in weird]))
    assert got["properties.happiness"] == OpenSearchSchemaField(type="str", samples={"yes"})

    assert "properties.entity.location" not in got
