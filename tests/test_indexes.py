# The SPECTER2 encoder (BUILD_PLAN.md Day 3, step 3.2). Skips when the model is not
# downloaded (python scripts/download_all.py).
import numpy as np
import pytest

from gea import config as cfg
from gea.indexes import embed_text, specter

QUESTION = "Which dataset has the most users?"
ON_TOPIC = "MovieLens-1M is the largest of the three datasets, with 6,040 users and one million ratings."
UNRELATED = "The learning rate is set to 0.001 and decayed every ten epochs."


@pytest.fixture(scope="module", autouse=True)
def loaded():
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
