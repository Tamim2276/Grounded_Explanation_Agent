# The notebook's checks, runnable without Jupyter:   pytest -q
#
# Every test runs on the stub backend (the fake two-page paper), so none of them
# needs a GPU, a model or the network. Each name says which notebook test it mirrors.
from collections import defaultdict

import pymupdf
import pytest
from langgraph.graph import END, START, StateGraph

from gea import config as cfg
from gea.corpus import get_corpus
from gea.graph import (build_gated_graph, build_grounded_agent_graph, build_query_graph,
                       build_routing_graph)
from gea.profiling import (EXPECTED_COSTS, SCENARIOS, predicted_steps, profile, run_scenario,
                           run_with_style)
from gea.proofs import iou
from gea.question import n1_query_intake, n2_query_encoder
from gea.planner import n5_agentic_planner
from gea.state import AgentState, new_state
from gea.stub_paper import (FAKE_PAPER, FIGURE_QUERY, PAPER_ID, QUERY, TABLE_QUERY, TEXT_QUERY,
                            UNANSWERABLE)
from gea.viz import render_paper


@pytest.fixture(autouse=True)
def quiet():
    # No trace lines, and the stub backend, for every test.
    with cfg.override(TRACE=False, BACKEND="stub"):
        yield


@pytest.fixture(scope="module")
def agent():
    return build_grounded_agent_graph()


def run(graph, question, **settings):
    with cfg.override(**settings):
        return graph.invoke(new_state(question))


# --- Section 5: the offline path ------------------------------------------------

def on_page(box):
    return 0 <= box[0] < box[2] <= cfg.PAGE_W and 0 <= box[1] < box[3] <= cfg.PAGE_H


def test_5a_indexes_built_and_every_box_on_its_page():
    corpus = get_corpus(PAPER_ID)
    records = corpus["chunks"] + corpus["tables"] + corpus["figures"]
    assert all(on_page(r["bbox"]) for r in records)
    assert all(on_page(b) for row in corpus["tables"][0]["cell_boxes"] for b in row)
    assert corpus["page_index"]["patches"].shape == (2, 32 * 32, cfg.COLPALI_DIM)


def test_5b_all_row_is_the_weighted_average():
    t3 = FAKE_PAPER["tables"][0]
    rows = {r[0]: [float(v) for v in r[1:]] for r in t3["rows"]}
    for col in (0, 1):
        avg = sum(w * rows[s][col] for s, w in FAKE_PAPER["subset_weights"].items())
        assert round(avg, 1) == rows["All"][col]
    assert round(rows["All"][1] - rows["All"][0], 1) == rows["All"][2] == 3.9


# --- Sections 6-7: the query path and routing ------------------------------------

def test_6_linear_query_path_buffers_the_right_paragraph():
    out = build_query_graph().invoke(new_state(TEXT_QUERY))
    assert [e.source for e in out["evidence"]] == ["c3"]
    assert out["new_evidence"] == []


def test_6c_n1_to_n5_edge_runs_the_planner_too_early():
    g = StateGraph(AgentState)
    g.add_node("n1_query_intake", n1_query_intake)
    g.add_node("n2_query_encoder", n2_query_encoder)
    g.add_node("n5_agentic_planner", n5_agentic_planner)
    g.add_edge(START, "n1_query_intake")
    g.add_edge("n1_query_intake", "n2_query_encoder")
    g.add_edge("n1_query_intake", "n5_agentic_planner")
    g.add_edge("n2_query_encoder", "n5_agentic_planner")
    g.add_edge("n5_agentic_planner", END)
    with pytest.raises(AssertionError, match="N2 has not run"):
        g.compile().invoke(new_state(QUERY))


@pytest.mark.parametrize("question, kind, source", [
    (TEXT_QUERY, "text", "c3"), (TABLE_QUERY, "table", "t3"), (FIGURE_QUERY, "figure", "f2")])
def test_7a_each_question_reaches_its_tool(question, kind, source):
    out = build_routing_graph().invoke(new_state(question))
    assert (out["evidence"][0].kind, out["evidence"][0].source) == (kind, source)


def test_7b_zoom_depends_on_figure_size_only():
    graph = build_routing_graph()
    zoomed = run(graph, FIGURE_QUERY, ZOOM_IF_AREA_BELOW=0.30)
    direct = run(graph, FIGURE_QUERY, ZOOM_IF_AREA_BELOW=0.05)
    assert [e.tool for e in zoomed["evidence"]] == ["FigureRetriever", "RegionZoom"]
    assert [e.tool for e in direct["evidence"]] == ["FigureRetriever"]


# --- Section 8: G1 and the retrieval loop ------------------------------------------

@pytest.mark.parametrize("question, status, rounds", [
    (TEXT_QUERY, "ok", 1), (QUERY, "ok", 2), (UNANSWERABLE, "insufficient_evidence", cfg.MAX_ROUNDS)])
def test_8_gate_loops_with_a_bound_and_abstains(question, status, rounds):
    out = build_gated_graph().invoke(new_state(question))
    assert (out["status"], out["round"]) == (status, rounds)


def test_8d_claim_linked_to_the_table_it_cites():
    out = build_gated_graph().invoke(new_state(QUERY))
    assert ("E1", "E2", "same page, text cites Table 3") in out["links"]


# --- Section 9: N9, G2 and the regenerate loop ----------------------------------------

def test_9a_honest_draft_passes_untouched(agent):
    out = agent.invoke(new_state(QUERY))
    assert out["faith_failures"] == [] and out["rewrite_count"] == 0 and out["status"] == "ok"


def test_9b_misquote_caught_and_rewritten(agent):
    out = run(agent, QUERY, HALLUCINATE_ONCE=True)
    assert out["rewrite_count"] == 1 and out["faith_failures"] == []
    assert any("Baseline 71.3" in c["text"] for c in out["claims"])


def test_9c_never_corrected_claim_is_withheld(agent):
    out = run(agent, QUERY, HALLUCINATE_ALWAYS=True)
    withheld = [c["text"] for c in out["claims"] if c.get("rejected")]
    assert out["rewrite_count"] == cfg.MAX_REWRITES
    assert out["status"] == "unsupported_claims_dropped"
    assert len(withheld) == 1 and "72.3" in withheld[0]


# --- Section 10: proofs and the finished graph ---------------------------------------

def test_10d_running_example_boxes_every_cited_cell(agent):
    out = agent.invoke(new_state(QUERY))
    cells = {b["what"] for b in out["boxes"] if "cell" in b["what"]}
    assert {"Table 3 cell Low-resource / Gain", "Table 3 cell All / Gain"} <= cells
    assert out["dag_trace"]["g1_rejections"] == 1


def test_10e_figure_box_from_patch_map_lands_on_the_figure(agent):
    out = agent.invoke(new_state(FIGURE_QUERY))
    box = next(b["bbox"] for b in out["boxes"] if "patch map" in b["what"])
    assert iou(box, FAKE_PAPER["figures"][0]["bbox"]) >= 0.85


def test_10f_unanswerable_abstains_with_a_trace(agent):
    out = agent.invoke(new_state(UNANSWERABLE))
    assert out["status"] == "insufficient_evidence"
    assert out["claims"] == [] and "boxes" not in out
    assert out["dag_trace"]["path"].endswith("abstain") and "NO ANSWER" in out["report"]


def test_10g_only_n8_writes_the_buffer(agent):
    writers = defaultdict(set)
    for q in (QUERY, FIGURE_QUERY, UNANSWERABLE):
        for event in agent.stream(new_state(q), stream_mode="updates"):
            for node, update in event.items():
                for key in (update or {}):
                    writers[key].add(node)
    assert writers["evidence"] == {"n8_evidence_buffer"}
    assert writers["claims"] == {"n9_grounded_generator"}


def test_10h_drawn_wiring_matches_the_thesis(agent):
    edges = {(e.source, e.target) for e in agent.get_graph().edges}
    assert len(edges) == 32
    assert ("replan_prep", "n5_agentic_planner") in edges
    assert ("rewrite_prep", "n9_grounded_generator") in edges
    assert ("abstain", "n10c_dag_trace") in edges
    assert ("n1_query_intake", "n5_agentic_planner") not in edges


# --- Section 11: costs ---------------------------------------------------------------

def test_11c_settings_restored_after_a_crash(agent):
    before = (cfg.TRACE, cfg.HALLUCINATE_ONCE, cfg.HALLUCINATE_ALWAYS)
    with pytest.raises(AssertionError):
        run_scenario(agent, None, lie_always=True)
    assert (cfg.TRACE, cfg.HALLUCINATE_ONCE, cfg.HALLUCINATE_ALWAYS) == before


def test_11d_costs_unchanged_and_predicted_by_the_formula(agent):
    for sc in SCENARIOS:
        p = profile(run_scenario(agent, sc["question"], lie_once=sc.get("lie_once", False),
                                 lie_always=sc.get("lie_always", False)))
        assert (p["steps"], p["node_runs"], p["model_calls"]) == EXPECTED_COSTS[sc["name"]], sc["name"]
        assert predicted_steps(p["rounds"], p["zooms"], p["rewrites"], p["dropped"], p["abstained"]) \
            == p["steps"], sc["name"]


# --- Section 13: proofs checked against a real PDF ------------------------------------

def test_13a_every_boxed_cell_shows_its_cited_value_in_the_pdf(agent):
    out = agent.invoke(new_state(QUERY))
    page = render_paper(FAKE_PAPER)[1]
    for b in out["boxes"]:
        if "cell" not in b["what"]:
            continue
        claim = out["claims"][b["claim"] - 1]
        cited = next(c["value"] for c in claim["cites"]
                     if f"{c['cell'][0]} / {c['cell'][1]}" in b["what"])
        assert page.get_textbox(pymupdf.Rect(b["bbox"])).strip() == cited


# --- Section 14: the LLM planner falls back to the rules -------------------------------

def test_14c_unreachable_server_falls_back_to_rules(agent):
    out = run(agent, QUERY, PLANNER_MODE="llm", LLM_URL="http://localhost:9/v1")
    assert out["action"]["planner"] == "rules (LLM unavailable)"
    assert [e.source for e in out["evidence"]] == ["c4", "t3"]


# --- Section 15: where G2 stops protecting ---------------------------------------------

def test_15a_g2_catches_misquotes_but_not_misattribution(agent):
    honest = run_with_style(agent, "honest")
    fabricated = run_with_style(agent, "honest", lie_once=True)
    uncited = run_with_style(agent, "uncited")
    misattributed = run_with_style(agent, "misattributed")
    assert honest["rewrite_count"] == 0
    assert fabricated["rewrite_count"] == 1
    assert uncited["status"] == "unsupported_claims_dropped"
    assert misattributed["rewrite_count"] == 0                 # the boundary: it passes
    claim = next(c for c in misattributed["claims"] if c.get("conclusion"))
    assert claim["cites"][0]["cell"] == ("High-resource", "Gain") and "low-resource" in claim["text"]
    assert cfg.TABLE_CLAIM_STYLE == "honest"
