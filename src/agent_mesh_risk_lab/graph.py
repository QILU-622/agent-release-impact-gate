"""Agent mesh graph construction."""

from __future__ import annotations

import networkx as nx

from .catalog import TOOLS, WORKFLOWS


def build_workflow_graph(workflow: str) -> nx.DiGraph:
    spec = WORKFLOWS[workflow]
    graph = nx.DiGraph(workflow=workflow)
    for node in spec.chain:
        graph.add_node(node, node_type="tool" if node.endswith("Tool") else "agent")
    for source, target in zip(spec.chain, spec.chain[1:]):
        relation = "calls" if target.endswith("Tool") else "delegates_to"
        graph.add_edge(source, target, relation=relation, risk_level="medium")
    for tool_name in spec.tools:
        tool = TOOLS[tool_name]
        graph.add_node(tool_name, node_type="tool", risk_level=tool.risk_level)
        caller = next(
            (node for node in reversed(spec.chain) if not node.endswith("Tool")), spec.chain[0]
        )
        graph.add_edge(caller, tool_name, relation="can_call", risk_level=tool.risk_level)
    for index, _policy in enumerate(spec.policies, start=1):
        policy_id = f"P{index}"
        graph.add_node(policy_id, node_type="policy")
        graph.add_edge(policy_id, spec.chain[0], relation="must_follow", risk_level="low")
    return graph


def graph_summary(workflow: str) -> dict[str, float]:
    graph = build_workflow_graph(workflow)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "delegation_depth": nx.dag_longest_path_length(graph),
        "density": round(nx.density(graph), 4),
    }
