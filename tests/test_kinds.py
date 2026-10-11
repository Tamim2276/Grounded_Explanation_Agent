# N1's kind of evidence by the vote of labelled SPIQA questions (gea/kinds.py). The rule on
# made-up vectors first; then the real examples, which must never include a dev or test
# question; then N1 using the vote on the real backend only.
import numpy as np
import pytest

import gea.kinds as kinds
from gea import config as cfg
from gea.question import n1_query_intake
from gea.safeio import read_jsonl
from gea.state import new_state


def test_the_majority_of_the_nearest_examples_decides():
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]])
    labels = np.array(["table", "table", "figure", "figure", "figure"])
    assert kinds.vote(np.array([1.0, 0.0]), vectors, labels, k=3) == ("table", 2)
    assert kinds.vote(np.array([0.0, 1.0]), vectors, labels, k=3) == ("figure", 3)


def test_no_dev_or_test_question_is_an_example():
    if not kinds.SPIQA_JSON.exists():
        pytest.skip("SPIQA test-A not downloaded (python scripts/download_all.py)")
    questions, labels = kinds.example_questions(kinds.evaluated_papers())
    evaluated = {q["question"] for split in ("dev", "test")
                 for q in read_jsonl(cfg.PROJECT_ROOT / "eval" / "splits" / f"{split}.jsonl")}
    assert len(questions) == 469 and not evaluated & set(questions)
    assert sorted(set(labels)) == ["figure", "table"]


def test_n1_takes_the_voted_kind_on_real_papers_only(monkeypatch):
    monkeypatch.setattr(kinds, "kind_of", lambda text: ("table", 7, 9))
    q = "Which method performed the best on the GRID dataset?"
    with cfg.override(BACKEND="real", LLM_MODE="rules", TRACE=False):
        real = n1_query_intake(new_state(q))
    assert [g["modality"] for g in real["sub_goals"]] == ["table"]
    assert "kinds by vote 7/9 table" in real["events"][0]["msg"]
    with cfg.override(BACKEND="stub", TRACE=False):                   # the fake paper keeps the rules
        stub = n1_query_intake(new_state(q))
    assert [g["modality"] for g in stub["sub_goals"]] == ["text"]
    with cfg.override(BACKEND="real", LLM_MODE="rules", KIND_ROUTER="split", TRACE=False):
        split = n1_query_intake(new_state(q))
    assert [g["modality"] for g in split["sub_goals"]] == ["text"]
