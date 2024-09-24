from typing import Any, Set

from sycamore.query.logical_plan import LogicalPlan, Node


def compare_graphs(node_a: Node, node_b: Node, visited_a: Set[int], visited_b: Set[int]) -> dict[str, Any]:
    """
    Traverse and compare 2 graphs given a node pointer in each. Computes different comparison metrics per node.
    The function will continue to traverse as long as the graph structure is identical, i.e. same number of outgoing
    nodes per node. It also assumes that the "downstream_nodes"/edges are ordered - this is the current logical
    plan implementation to support operations like math.


    @param node_a: graph node a
    @param node_b: graph node b
    @param visited_a: helper to track traversal in graph a
    @param visited_b: helper to track traversal in graph b
    @return: Dict with comparison metrics in the format diff_type_name: [(node_a, node_b)...]
    """
    diff_result: dict[str, Any] = {
        "node_type_diff_result": [],
        "node_data_diff_result": [],
        "structural_diff_result": [],
    }

    if node_a.node_id in visited_a and node_b.node_id in visited_b:
        return diff_result

    visited_a.add(node_a.node_id)
    visited_b.add(node_b.node_id)

    # Compare node types
    if type(node_a) != type(node_b):
        diff_result["node_type_diff_result"].append((node_a, node_b))

    # Compare node data
    if not node_a.logical_compare(node_b):
        diff_result["node_data_diff_result"].append((node_a, node_b))

    # Compare the structure (inputs)
    if len(node_a._downstream_nodes) != len(node_b._downstream_nodes):
        diff_result["structural_diff_result"].append((node_a, node_b))
    else:
        for input1, input2 in zip(node_a._downstream_nodes, node_b._downstream_nodes):
            sub_diff_result = compare_graphs(input1, input2, visited_a, visited_b)
            for key in diff_result:
                diff_result[key].extend(sub_diff_result[key])

    return diff_result


def compare_logical_plans_from_query_source(plan_a: LogicalPlan, plan_b: LogicalPlan) -> dict[str, Any]:
    """
    A simple method to compare 2 logical plans. This comparator traverses a plan 'forward', i.e. it attempts to
    start from node_id == 0 which is typically a data source query. This helps us detect differences in the plan
    in the natural flow of data. If the plans diverge structurally, i.e. 2 nodes have different number of downstream
    nodes we stop traversing.


    @param plan_a: Plan a
    @param plan_b: Plan b
    @return: Dict with comparison metrics in the format diff_type_name: [(plan_a, plan_b)...]
    """
    assert 0 in plan_a.nodes, "Plan a requires at least 1 node indexed [0]"
    assert 0 in plan_b.nodes, "Plan b requires at least 1 node indexed [0]"
    return compare_graphs(plan_a.nodes[0], plan_b.nodes[0], set(), set())
