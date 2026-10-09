# The SPECTER2 encoder and the text index (BUILD_PLAN.md Day 3, steps 3.2-3.3). Skips
# when the model is not downloaded (python scripts/download_all.py).
import json
import os

import numpy as np
import pytest

from gea import config as cfg
from gea.device import free_memory_gb
from gea.indexes import (TEXT_INDEX, build_text_index, embed_text, load_text_index, specter,
                         text_index_is_current, text_records)

QUESTION = "Which dataset has the most users?"
ON_TOPIC = "MovieLens-1M is the largest of the three datasets, with 6,040 users and one million ratings."
UNRELATED = "The learning rate is set to 0.001 and decayed every ten epochs."


@pytest.fixture(scope="module", autouse=True)
def loaded():
    # Loading SPECTER2 takes ~1 GB at once; with less free, skip rather than crash
    # whatever else is running on this PC.
    if free_memory_gb() < 2.5:
        pytest.skip(f"only {free_memory_gb():.1f} GB of memory free; SPECTER2 tests skipped")
    try:
        specter()
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_one_unit_length_vector_per_text():
    v = embed_text([ON_TOPIC, UNRELATED, "accuracy " * 600])   # the last is cut at 512 tokens
    assert v.shape == (3, cfg.SPECTER2_DIM) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_the_same_text_gets_the_same_vector():
    assert float(embed_text([ON_TOPIC])[0] @ embed_text([ON_TOPIC])[0]) == pytest.approx(1.0, abs=1e-5)


def test_batching_and_padding_do_not_change_a_vector():
    # 20 texts of different lengths = one batch of 16 + one of 4, padded to the longest
    texts = [f"Table {i}: results" + " on the test set" * i for i in range(20)]
    together = embed_text(texts)
    alone = np.concatenate([embed_text([t]) for t in texts])
    assert together.shape == (20, cfg.SPECTER2_DIM)
    assert np.allclose(together, alone, atol=1e-4)


def test_the_two_adapters_are_different_hats():
    assert float(embed_text([QUESTION], "doc")[0] @ embed_text([QUESTION], "query")[0]) < 0.99


def test_an_on_topic_paragraph_beats_an_unrelated_one():
    q = embed_text([QUESTION], kind="query")[0]
    on_topic, unrelated = embed_text([ON_TOPIC, UNRELATED])
    assert q @ on_topic > q @ unrelated


def test_unknown_kind_and_empty_input():
    with pytest.raises(ValueError):
        embed_text(["x"], kind="question")
    assert embed_text([]).shape == (0, cfg.SPECTER2_DIM)


def search_own_texts(folder, records):
    # Search the index with each record's own text: (ids found first, their scores).
    idx = load_text_index(folder)
    scores, rows = idx["faiss"].search(embed_text([t for _, t in records]), 1)
    return [idx["ids"][r] for r in rows[:, 0]], scores[:, 0]


def tiny_paper(folder) -> list:
    paper = {"chunks": [{"id": "c1", "text": "MovieLens-1M has 6,040 users and one million ratings."},
                        {"id": "c2", "text": "The learning rate is set to 0.001."}],
             "tables": [{"id": "t1", "caption": "Table 1: Statistics of the three datasets."}],
             "figures": [{"id": "f1", "caption": "Figure 1: Training loss per epoch."}]}
    (folder / "corpus.json").write_text(json.dumps(paper), encoding="utf-8")
    return text_records(paper)


def test_a_text_index_finds_each_record_by_its_own_text(tmp_path):
    records = tiny_paper(tmp_path)
    assert build_text_index(tmp_path) == 4
    assert load_text_index(tmp_path)["ids"] == ["c1", "c2", "t1", "f1"]   # chunks, then captions
    found, scores = search_own_texts(tmp_path, records)
    assert found == ["c1", "c2", "t1", "f1"]
    assert np.allclose(scores, 1.0, atol=1e-4)


def test_an_index_older_than_its_paper_is_refused(tmp_path):
    tiny_paper(tmp_path)
    build_text_index(tmp_path)
    later = (tmp_path / TEXT_INDEX).stat().st_mtime + 10
    os.utime(tmp_path / "corpus.json", (later, later))          # N3 read the paper again
    assert not text_index_is_current(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_text_index(tmp_path)


def test_a_power_cut_before_the_index_leaves_the_paper_undone(tmp_path):
    tiny_paper(tmp_path)
    (tmp_path / "text_ids.json").write_text('{"ids": ["c1"]}', encoding="utf-8")   # ids saved, index not
    assert not text_index_is_current(tmp_path)


def test_a_real_paper_index_matches_its_records():
    folder = cfg.DATA_DIR / "corpus" / "1803.03467v4"
    if not (folder / "corpus.json").exists() or not text_index_is_current(folder):
        pytest.skip("run: python scripts/build_indexes.py text")
    records = text_records(json.loads((folder / "corpus.json").read_text(encoding="utf-8")))
    assert load_text_index(folder)["ids"] == [i for i, _ in records]
    found, scores = search_own_texts(folder, records[:10])
    assert found == [i for i, _ in records[:10]]
    assert np.allclose(scores, 1.0, atol=1e-4)
