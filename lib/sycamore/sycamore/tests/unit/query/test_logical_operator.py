from hashlib import sha256
import json

from sycamore.query.logical_plan import Node
from sycamore.query.operators.count import Count
from sycamore.query.operators.query_database import QueryDatabase


def test_node_cache_dict():
    node1 = Node(node_id=1)
    assert node1.cache_dict() == {"operator_type": "Node", "dependencies": []}
    assert node1.cache_key() == sha256(json.dumps(node1.cache_dict()).encode()).hexdigest()

    # Changing node ID does not affect cache_dict.
    node2 = Node(node_id=2)
    assert node2.cache_dict() == node1.cache_dict()
    assert node2.cache_key() == node1.cache_key()

    # QueryDatabase node.
    node3 = QueryDatabase(node_id=3, description="Test description", index="ntsb", query={"match_all": {}})
    assert node3.cache_dict() == {
        "operator_type": "QueryDatabase",
        "dependencies": [],
        "index": "ntsb",
        "query": {"match_all": {}},
    }
    assert node3.cache_key() == sha256(json.dumps(node3.cache_dict()).encode()).hexdigest()

    # Count node that depends on QueryDatabase node.
    node4 = Count(node_id=4, description="Count description", distinct_field="temperature")
    # pylint: disable=protected-access
    node4._dependencies = [node3]
    assert node4.cache_dict() == {
        "operator_type": "Count",
        "dependencies": [
            {
                "operator_type": "QueryDatabase",
                "dependencies": [],
                "index": "ntsb",
                "query": {"match_all": {}},
            }
        ],
        "distinct_field": "temperature",
    }
    assert node4.cache_key() == sha256(json.dumps(node4.cache_dict()).encode()).hexdigest()
