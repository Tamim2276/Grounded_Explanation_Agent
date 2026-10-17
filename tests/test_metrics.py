# The Day 8 scores on the toy examples of BUILD_PLAN.md (Day 8, "Understand it first").
import pytest

from gea.metrics import (FLAG, aes, answer_text, bootstrap_diff, bootstrap_mean, cgs, cohen_kappa, outcome,
                         published, rf, search_calls, visual_labels)


def record(claims=(), answer="", evidence=(), path=""):
    return {"claims": [{"rejected": r} for r in claims], "answer": answer, "evidence": list(evidence),
            "dag_path": path}


def test_rf_toy_example():
    assert rf(["Table 3", "Figure 1", "Figure 2"], "Table 3") == pytest.approx(1 / 3)
    assert rf([], "Table 3") is None                       # nothing visual fetched: undefined, not 0


def test_rf_counts_a_zoom_as_its_figure():
    ev = [{"kind": "figure", "label": "Figure 2"}, {"kind": "figure", "label": "Figure 2"},   # + its zoom
          {"kind": "table", "label": "Table 3"}, {"kind": "text", "label": "Section 4 paragraph"}]
    assert visual_labels(record(evidence=ev)) == {"Figure 2", "Table 3"}


def test_aes_toy_example():
    assert aes(2, 4) == 0.5
    assert aes(2, 1) == 1.0                                # capped at 1
    assert aes(1, 0) == 0.0


def test_search_calls_leave_out_the_automatic_zoom():
    path = "TextRetriever -> G1 reject -> FigureRetriever -> RegionZoom -> G1 accept -> draft 1"
    assert search_calls(record(path=path)) == 2
    assert search_calls(record(path=None)) == 0


def test_cgs_toy_example():
    # two 10x10 boxes overlapping by half: 50 / 150
    assert cgs([{"page": 3, "bbox": [0, 0, 10, 10]}], [(3, (5, 0, 15, 10))]) == pytest.approx(1 / 3)


def test_cgs_takes_the_best_box_on_the_same_page_and_0_without_one():
    boxes = [{"page": 3, "bbox": [0, 0, 10, 10]}, {"page": 4, "bbox": [5, 0, 15, 10]}]
    assert cgs(boxes, [(4, (5, 0, 15, 10))]) == 1.0
    assert cgs(boxes, [(5, (5, 0, 15, 10))]) == 0.0        # nothing on the gold page
    assert cgs([], [(3, (0, 0, 10, 10))]) == 0.0            # an abstention


def test_outcome_three_ways():
    assert outcome(record(claims=[False, True]), True) == "correct"
    assert outcome(record(claims=[False]), False) == "wrong"
    assert outcome(record(claims=[True, True]), None) == "abstained"   # every claim withheld by G2
    assert outcome(record(), None) == "abstained"
    assert published(record(claims=[True, False]))


def test_answer_text_drops_the_best_effort_mark():
    assert answer_text(record(answer="Model A scores 63.1." + FLAG)) == "Model A scores 63.1."
    assert answer_text({"answer": None}) == ""


def test_bootstrap_sees_a_real_difference_and_a_tie():
    a, b = [1] * 80 + [0] * 70, [1] * 40 + [0] * 110
    diff, (low, high) = bootstrap_diff(a, b)
    assert diff == pytest.approx(40 / 150) and 0 < low < diff < high
    _, (low, high) = bootstrap_diff(a, a)
    assert low == high == 0
    mean, (low, high) = bootstrap_mean([1, 0] * 75)
    assert mean == 0.5 and low < 0.5 < high


def test_cohen_kappa():
    assert cohen_kappa(["c", "w", "c", "w"], ["c", "w", "c", "w"]) == 1.0
    # observed 0.5, expected 0.5 -> no better than chance
    assert cohen_kappa(["c", "c", "w", "w"], ["c", "w", "c", "w"]) == 0.0
