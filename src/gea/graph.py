# The graph builders, in the order the notebook grows the graph: a straight line,
# then modality routing, then G1 with the retrieval loop, then N9/G2 with the
# regenerate loop, then the full graph of thesis Figure 4.1.
#
# A node is registered under the same name as the function that implements it, so
# a box in show_graph() can be found in the code by searching for its label.
from langgraph.graph import END, START, StateGraph

from gea.generation import (drop_unsupported, faithfulness_gate, g2_faithfulness_check,
                            gate_until_proofs, n9_grounded_generator, rewrite_prep)
from gea.planner import n5_agentic_planner
from gea.proofs import (n10a_claim_links, n10b_bounding_boxes, n10c_dag_trace,
                        n11_report_synthesis, n12_final_output)
from gea.question import n1_query_intake, n2_query_encoder
from gea.retrieval import (TOOL_NODES, n6a_text_retriever, n6b_table_retriever,
                           n6c_figure_retriever, n7_region_zoom, n8_evidence_buffer,
                           route_tool, route_zoom)
from gea.state import AgentState
from gea.sufficiency import abstain, g1_sufficiency_check, replan_prep, sufficiency_gate


def add_retrieval(g: StateGraph) -> None:
    # Shared by every builder from Step 2 on: N1 -> N2 -> N5 -> one tool -> N8.
    for name, fn in [("n1_query_intake", n1_query_intake),
                     ("n2_query_encoder", n2_query_encoder),
                     ("n5_agentic_planner", n5_agentic_planner),
                     ("n6a_text_retriever", n6a_text_retriever),
                     ("n6b_table_retriever", n6b_table_retriever),
                     ("n6c_figure_retriever", n6c_figure_retriever),
                     ("n7_region_zoom", n7_region_zoom),
                     ("n8_evidence_buffer", n8_evidence_buffer)]:
        g.add_node(name, fn)
    g.add_edge(START, "n1_query_intake")
    g.add_edge("n1_query_intake", "n2_query_encoder")
    g.add_edge("n2_query_encoder", "n5_agentic_planner")
    g.add_conditional_edges("n5_agentic_planner", route_tool, TOOL_NODES)    # one edge, three tools
    g.add_conditional_edges("n6c_figure_retriever", route_zoom, {
        "zoom": "n7_region_zoom", "buffer": "n8_evidence_buffer"})
    g.add_edge("n6a_text_retriever", "n8_evidence_buffer")
    g.add_edge("n6b_table_retriever", "n8_evidence_buffer")
    g.add_edge("n7_region_zoom", "n8_evidence_buffer")


def add_g1_loop(g: StateGraph, on_sufficient: str) -> None:
    # G1, the bounded retrieval loop and the abstain exit.
    g.add_node("g1_sufficiency_check", g1_sufficiency_check)
    g.add_node("replan_prep", replan_prep)
    g.add_node("abstain", abstain)
    g.add_edge("n8_evidence_buffer", "g1_sufficiency_check")
    g.add_conditional_edges("g1_sufficiency_check", sufficiency_gate, {   # the red diamond
        "sufficient":     on_sufficient,
        "retrieve_again": "replan_prep",
        "exhausted":      "abstain",
    })
    g.add_edge("replan_prep", "n5_agentic_planner")                         # the red dashed loop


def add_generation(g: StateGraph) -> None:
    # N9, G2's check node and the two loop helpers. The caller wires G2's routing.
    g.add_node("n9_grounded_generator", n9_grounded_generator)
    g.add_node("g2_faithfulness_check", g2_faithfulness_check)
    g.add_node("rewrite_prep", rewrite_prep)
    g.add_node("drop_unsupported", drop_unsupported)
    g.add_edge("n9_grounded_generator", "g2_faithfulness_check")
    g.add_edge("rewrite_prep", "n9_grounded_generator")                     # the regenerate loop


def build_query_graph():
    # Step 1 graph: the query path, linear -- one text retrieval into the buffer.
    g = StateGraph(AgentState)

    g.add_node("n1_query_intake",    n1_query_intake)
    g.add_node("n2_query_encoder",   n2_query_encoder)
    g.add_node("n5_agentic_planner", n5_agentic_planner)
    g.add_node("n6a_text_retriever", n6a_text_retriever)
    g.add_node("n8_evidence_buffer", n8_evidence_buffer)

    g.add_edge(START, "n1_query_intake")
    g.add_edge("n1_query_intake",    "n2_query_encoder")
    g.add_edge("n2_query_encoder",   "n5_agentic_planner")
    g.add_edge("n5_agentic_planner", "n6a_text_retriever")
    g.add_edge("n6a_text_retriever", "n8_evidence_buffer")
    g.add_edge("n8_evidence_buffer", END)

    return g.compile()


def build_routing_graph():
    # Step 2 graph: the planner chooses among three tools; small figures are zoomed.
    g = StateGraph(AgentState)
    add_retrieval(g)
    g.add_edge("n8_evidence_buffer", END)
    return g.compile()


def build_gated_graph():
    # Step 3 graph: routing + G1 + the bounded retrieval loop.
    g = StateGraph(AgentState)
    add_retrieval(g)
    add_g1_loop(g, on_sufficient=END)
    g.add_edge("abstain", END)
    return g.compile()


def build_generation_graph():
    # Step 4 graph: everything so far, plus N9, G2 and the bounded regenerate loop.
    g = StateGraph(AgentState)
    add_retrieval(g)
    add_g1_loop(g, on_sufficient="n9_grounded_generator")
    g.add_edge("abstain", END)
    add_generation(g)
    g.add_conditional_edges("g2_faithfulness_check", gate_until_proofs, {
        "faithful":  END,
        "rewrite":   "rewrite_prep",
        "exhausted": "drop_unsupported",
    })
    g.add_edge("drop_unsupported", END)
    return g.compile()


def build_grounded_agent_graph():
    # The complete query-time graph of thesis Chapter 4: every node, both gates,
    # both loops and both exits. (N3-N4 run offline, in corpus.build_ingest_graph.)
    g = StateGraph(AgentState)
    add_retrieval(g)
    add_g1_loop(g, on_sufficient="n9_grounded_generator")
    g.add_edge("abstain", "n10c_dag_trace")          # an abstention still gets its trace
    add_generation(g)

    # NOT deferred: N10a-N10c are siblings of one fan-out, so they finish in the
    # same step and N11 is triggered once, after all of them.
    g.add_node("n10a_claim_links",     n10a_claim_links)
    g.add_node("n10b_bounding_boxes",  n10b_bounding_boxes)
    g.add_node("n10c_dag_trace",       n10c_dag_trace)
    g.add_node("n11_report_synthesis", n11_report_synthesis)
    g.add_node("n12_final_output",     n12_final_output)

    g.add_conditional_edges("g2_faithfulness_check", faithfulness_gate, {
        "links":     "n10a_claim_links",
        "boxes":     "n10b_bounding_boxes",
        "trace":     "n10c_dag_trace",
        "rewrite":   "rewrite_prep",
        "exhausted": "drop_unsupported",
    })
    for proof in ("n10a_claim_links", "n10b_bounding_boxes", "n10c_dag_trace"):
        g.add_edge("drop_unsupported", proof)
        g.add_edge(proof, "n11_report_synthesis")

    g.add_edge("n11_report_synthesis", "n12_final_output")
    g.add_edge("n12_final_output", END)
    return g.compile()
