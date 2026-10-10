# Day 4 (BUILD_PLAN.md): table cells, region choice, heat-map boxes and the zoom.
# Rules on small hand-made inputs first; real papers last, skipped when the data or the
# memory is not there.
import json

import numpy as np
import pytest

from gea import config as cfg
from gea.device import free_memory_gb
from gea.indexes import specter_loaded
from gea.proofs import iou, patches_to_bbox
from gea.retrieval import inside_mask, masked_heat, region_heat, route_zoom, text_scores
from gea.state import Evidence
from gea.tables import cell_text, distinct, load_tables

CORPUS = cfg.DATA_DIR / "corpus"


# --- table structure -------------------------------------------------------------

def test_a_word_belongs_to_the_cell_holding_its_centre():
    # (x0, y0, x1, y1, text): one row's words, and the next row's word just below
    words = [(10, 10, 40, 20, "6,322,548"), (10, 19, 40, 29, "next-row")]
    assert cell_text(words, (0, 9, 50, 21)) == "6,322,548"       # touches next-row, does not own it


def test_words_on_slightly_different_baselines_stay_in_reading_order():
    words = [(42, 11, 60, 21, "et"), (62, 11, 70, 21, "al."), (10, 10, 40, 20, "Chung")]
    assert cell_text(words, (0, 0, 100, 30)) == "Chung et al."


def test_a_doubled_row_box_is_dropped():
    rows = [(0, 10, 100, 20), (0, 11, 100, 21), (0, 22, 100, 32)]
    assert distinct(rows, axis=1) == [(0, 10, 100, 20), (0, 22, 100, 32)]


# --- region choice and heat maps ---------------------------------------------------

def test_a_search_that_finds_nothing_gives_no_credit():
    # Only t1 shares words with the query; SPECTER2 slightly prefers t5. Without the
    # rule, the four non-matches tie for 2nd place in the keyword ranking and t5 wins.
    meaning = {"t1": 0.721, "t2": 0.733, "t5": 0.737}
    words = {"t1": 2.34, "t2": 0.0, "t5": 0.0}
    fused = text_scores(meaning, words, "hybrid")
    assert max(fused, key=fused.get) == "t1"


def test_region_heat_and_the_masked_map():
    heat = np.full((4, 4), 0.2)
    heat[0, 3] = 0.9                                  # hot outside the region (top right)
    heat[2, 1] = 0.6                                  # warm inside it
    region, size = (0, 50, 50, 100), (100, 100)      # bottom-left quarter of a 100 x 100 page
    assert inside_mask(heat, region, size).sum() == 4
    assert region_heat(heat, region, size, top=1) == pytest.approx(0.6)
    box = patches_to_bbox(masked_heat(heat, region, size), page_size=size)
    assert box == (25.0, 50.0, 50.0, 75.0)            # the warm patch, not the hot one outside


def test_the_box_uses_the_real_page_size():
    heat = np.zeros((31, 24))
    heat[12, 12] = 1.0
    x0, y0, x1, y1 = patches_to_bbox(heat, page_size=(612.0, 792.0))     # US Letter
    assert (x0, x1) == pytest.approx((306.0, 331.5)) and y0 == pytest.approx(12 * 792 / 31, abs=0.1)


def test_zoom_decides_on_the_real_page_size():
    fig = Evidence("figure", "f1", 1, "Figure 1: x", (0, 0, 300, 200), "FigureRetriever",
                   page_size=(612.0, 792.0))
    assert route_zoom({"new_evidence": [fig]}) == "zoom"            # 60,000 / 484,704 pt^2 = 12%
    big = Evidence("figure", "f2", 1, "Figure 2: x", (0, 0, 600, 700), "FigureRetriever",
                   page_size=(612.0, 792.0))
    assert route_zoom({"new_evidence": [big]}) == "buffer"


# --- real papers -------------------------------------------------------------------

def paper_or_skip(pid: str) -> dict:
    path = CORPUS / pid / "corpus.json"
    if not path.exists():
        pytest.skip(f"{pid} not read (python scripts/build_corpus.py)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_table_1_cells_match_the_pdf():
    paper = paper_or_skip("1803.03467v4")
    t1 = next(t for t in paper["tables"] if t["label"] == "Table 1")
    parsed = load_tables(CORPUS / "1803.03467v4").get(t1["id"])
    if not parsed:
        pytest.skip("run: python scripts/build_indexes.py tables")
    assert parsed["method"] == "tatr"
    assert parsed["header"] == ["", "MovieLens-1M", "Book-Crossing", "Bing-News"]
    row = next(r for r in parsed["rows"] if r[0] == "# 4-hop triples")
    assert row == ["# 4-hop triples", "923,718", "71,628", "6,322,548"]


@pytest.fixture
def real():
    if not specter_loaded() and free_memory_gb() < 2.5:
        pytest.skip("less than 2.5 GB of memory free; SPECTER2 not loaded")
    paper_or_skip("1803.03467v4")
    from gea.corpus import get_corpus
    with cfg.override(BACKEND="real", TRACE=False, REGION_SEARCH="text"):
        yield get_corpus("1803.03467v4")


def test_n6b_returns_the_table_and_the_answer_cell(real):
    from gea.retrieval import table_retriever
    [t] = table_retriever(real, "Which dataset has the most 4-hop triples?")
    assert t.label == "Table 1" and t.page_size == (612.0, 792.0)
    cell = next(c for c in t.cells if (c["row"], c["col"]) == ("# 4-hop triples", "Bing-News"))
    assert cell["value"] == "6,322,548"
    assert iou(tuple(cell["bbox"]), (510.2, 172.7, 555.6, 183.0)) > 0.8


def test_n6c_takes_a_named_figure_and_n7_crops_it_sharp(real):
    from PIL import Image
    import io
    from gea.retrieval import figure_retriever, zoom_crop
    [f] = figure_retriever(real, "what does it show", ref="4")
    assert f.label == "Figure 4"
    png, (h, w, _) = zoom_crop(real, f)
    assert Image.open(io.BytesIO(png)).size == (w, h) and 900 <= max(w, h) <= cfg.ZOOM_MAX_PX + 2


def test_the_whole_graph_runs_on_a_real_table_question(real):
    from gea.graph import build_grounded_agent_graph
    from gea.state import new_state
    with cfg.override(PLANNER_MODE="rules", SUB_GOAL_MODALITY="table"):
        out = build_grounded_agent_graph().invoke(new_state("Which dataset has the most 4-hop triples?",
                                                            "1803.03467v4"))
    assert any(e.label == "Table 1" for e in out["evidence"])
    assert out["answer"] and out["boxes"]
