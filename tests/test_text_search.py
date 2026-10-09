# N6a on the real backend (BUILD_PLAN.md Day 3, step 3.5): the ranking rules on small
# hand-made inputs, then the real search on a real paper (skipped when the index is not
# built or memory is short).
import pytest

from gea import config as cfg
from gea.corpus import get_corpus
from gea.device import free_memory_gb
from gea.indexes import text_index_is_current
from gea.retrieval import in_section, n6a_text_retriever, ranks, text_retriever, text_scores

PAPER = "1804.07931v2"
QUESTION = "What is the relationship between clicks and impressions?"
ANSWER = "c13"          # "... clicks over all impressions. Obviously, S c is a subset of S."


def test_a_section_includes_its_subsections():
    assert [in_section({"section": s}, "2") for s in ("2", "2.2", "2.3.1", "20", "1")] == \
           [True, True, True, False, False]


def test_equal_scores_share_a_rank():
    assert ranks({"a": 3.0, "b": 3.0, "c": 1.0}) == {"a": 1, "b": 1, "c": 3}


def test_hybrid_prefers_what_both_searches_rank_high():
    meaning = {"a": 0.90, "b": 0.80, "c": 0.10}     # SPECTER2 order: a, b, c
    words = {"a": 0.0, "b": 5.0, "c": 4.0}          # keyword order:  b, c, a
    fused = text_scores(meaning, words, "hybrid")
    assert max(fused, key=fused.get) == "b"         # 2nd and 1st beats 1st and 3rd
    assert text_scores(meaning, words, "keyword") == words
    with pytest.raises(ValueError):
        text_scores(meaning, words, "bm25")


@pytest.fixture
def real():
    if not text_index_is_current(cfg.DATA_DIR / "corpus" / PAPER):
        pytest.skip("run: python scripts/build_indexes.py text")
    if free_memory_gb() < 2.5:
        pytest.skip("less than 2.5 GB of memory free; SPECTER2 not loaded")
    with cfg.override(BACKEND="real", TRACE=False):
        yield get_corpus(PAPER)


@pytest.mark.parametrize("how", ["keyword", "specter2", "hybrid"])
def test_the_answer_is_among_three_paragraphs(real, how):
    with cfg.override(TEXT_SEARCH=how):
        found = text_retriever(real, QUESTION)
    assert len(found) == cfg.TEXT_TOP_K == 3
    assert ANSWER in [e.source for e in found]
    assert all(e.kind == "text" and e.source.startswith("c") for e in found)    # no caption rows
    assert all(e.page >= 1 and len(e.bbox) == 4 for e in found)                # grounded


def test_the_section_filter_and_its_fallback(real):
    assert {e.section.split(".")[0] for e in text_retriever(real, QUESTION, section="2")} == {"2"}
    assert len(text_retriever(real, QUESTION, section="9")) == 3        # no section 9: whole paper


def test_the_unchanged_node_runs_on_the_real_backend(real):
    state = {"paper_id": PAPER, "action": {"tool": "text", "query": QUESTION},
             "query_filters": {"refs": {}, "entities": []}}
    found = n6a_text_retriever(state)["new_evidence"]
    assert ANSWER in [e.source for e in found]
