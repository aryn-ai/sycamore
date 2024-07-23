import matplotlib.pyplot as plt
import networkx as nx

from sycamore.query.logical_plan import LogicalPlan


def build_graph(plan: LogicalPlan):
    graph = nx.DiGraph()
    for node in plan.nodes().values():
        graph.add_node(node.node_id, description=f"{type(node).__name__}\n{node.description}")
        if node.dependencies:
            for dep in node.dependencies:
                graph.add_edge(dep.node_id, node.node_id)

    return graph


def visualize_plan(logical_plan: LogicalPlan):
    graph = build_graph(logical_plan)
    pos = nx.spring_layout(graph)
    labels = {node: f'{node}\n{data["description"]}' for node, data in graph.nodes(data=True)}
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=3000, node_color="skyblue", font_size=9, arrows=True)
    plt.show(block=False)
    plt.pause(1)
