# The three evaluated systems (BUILD_PLAN.md Day 7) on the stub paper: each runs, and each
# differs from the full graph only in what it switches off.
import pytest

from gea import config as cfg
from gea.graph import build_graph
from gea.state import new_state
from gea.stub_paper import QUERY, UNANSWERABLE


@pytest.fixture(autouse=True)
def quiet():
    with cfg.override(TRACE=False, BACKEND="stub"):
        yield


def run(variant: str, question: str = QUERY) -> dict:
    with cfg.override(VARIANT=variant):
        return build_graph().invoke(new_state(question))


def path(out: dict) -> list:
    return [e["short"] for e in out["events"] if e["short"]]


def test_the_full_system_is_unchanged():
    out = run("full")
    assert out["status"] == "ok" and out["boxes"]
    assert {"TableRetriever", "TextRetriever"} <= set(path(out))       # the running example's two tools


def test_single_pass_calls_every_tool_once_and_never_loops():
    out = run("single_pass")
    tools = [s for s in path(out) if s.endswith("Retriever")]
    assert sorted(tools) == ["FigureRetriever", "TableRetriever", "TextRetriever"]
    nodes = {e["node"] for e in out["events"]}
    assert not nodes & {"N5", "G1", "G2"}                               # no planner, no gates
    assert out["round"] == 1 and out.get("rewrite_count", 0) == 0
    assert out["answer"] and out["dag_trace"]["tool_calls"] >= 3


def test_single_pass_never_abstains():
    # The full graph abstains on this question; without G1 the baseline answers anyway.
    assert run("full", UNANSWERABLE)["status"] == "insufficient_evidence"
    assert run("single_pass", UNANSWERABLE)["status"] != "insufficient_evidence"


def test_text_only_never_calls_the_table_or_figure_tools():
    out = run("text_only")
    assert set(s for s in path(out) if s.endswith("Retriever")) == {"TextRetriever"}
    assert all(g["modality"] == "text" for g in out["sub_goals"])
    assert all(e.kind == "text" for e in out["evidence"])


def test_text_only_planner_is_offered_the_text_tool_alone():
    from gea.llm import TOOL_TEXT
    with cfg.override(VARIANT="text_only"):
        assert cfg.tools() == ("text",)
    assert ", ".join(TOOL_TEXT[t] for t in cfg.tools()) == \
        "'text' searches paragraphs, 'table' searches tables, 'figure' searches charts"   # full prompt unchanged


def test_text_only_reads_tables_and_figures_as_plain_text():
    from gea.retrieval import real_text_retriever
    path_ = cfg.DATA_DIR / "corpus" / "1803.03467v4" / "corpus.json"
    if not path_.exists():
        pytest.skip("1803.03467v4 not read")
    from gea.corpus import load_real_corpus
    corpus = load_real_corpus("1803.03467v4")
    query = "Which dataset has the most 4-hop triples?"
    with cfg.override(BACKEND="real", TEXT_SEARCH="keyword"):           # keyword: no SPECTER2 needed
        with cfg.override(VARIANT="full"):
            full = real_text_retriever(corpus, query, k=20)
        with cfg.override(VARIANT="text_only"):
            text_only = real_text_retriever(corpus, query, k=20)
    assert all(e.source.startswith("c") for e in full)                  # paragraphs only
    table = next(e for e in text_only if e.source == "t1")               # Table 1, as plain text (ranked
                                                                         # with the paragraphs, not first)
    assert table.kind == "text" and "6,322,548" in table.content and not table.cells


def test_text_only_may_search_again_with_a_new_query():
    # one tool: the no-repeat rule would make rounds 2 and 3 the same search (dev, Day 7)
    from gea.planner import n5_agentic_planner
    from gea.question import n1_query_intake, n2_query_encoder
    with cfg.override(VARIANT="text_only"):
        state = new_state(QUERY)
        state.update(n1_query_intake(state))
        state.update(n2_query_encoder(state))
        state["tried"] = [(0, "text")]
        state["sufficiency"] = {"sufficient": False, "uncovered": [0], "missing": "a number"}
        action = n5_agentic_planner(state)["action"]
    assert action["tool"] == "text" and "overruled" not in action["planner"]
