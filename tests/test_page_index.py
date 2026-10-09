# The page index (BUILD_PLAN.md Day 3, step 3.4) without loading ColQwen2: MaxSim on
# the plan's toy example, the room check, and the saved real pages when they exist.
import pytest
import torch

from gea import config as cfg
from gea.device import room_problems
from gea.indexes import load_page_index, maxsim, page_index_is_current


def test_maxsim_on_the_toy_example():
    # The plan's table: question "low-resource accuracy", a page of 4 patches. With the
    # two query tokens as the unit vectors, each patch's vector IS its column of scores.
    query = torch.eye(2)                                   # low-resource, accuracy
    page = torch.tensor([[0.1, 0.2],                       # P1 title
                         [0.9, 0.7],                       # P2 table row "Low-resource 71.3"
                         [0.2, 0.6],                       # P3 chart
                         [0.3, 0.4]])                      # P4 text
    other_page = torch.tensor([[0.3, 0.2], [0.1, 0.5]])
    assert maxsim(query, [page, other_page]) == pytest.approx([0.9 + 0.7, 0.3 + 0.5])


def test_the_room_check_says_why():
    assert room_problems(need_gb=0) == []
    [problem] = room_problems(need_gb=10**6)
    assert "memory is free" in problem


def test_every_saved_page_has_one_vector_per_grid_cell():
    folder = cfg.DATA_DIR / "corpus" / "1803.03467v4"
    if not (folder / "corpus.json").exists() or not page_index_is_current(folder):
        pytest.skip("run: python scripts/build_indexes.py pages")
    for p in load_page_index(folder):
        rows, cols = p["grid"]
        assert p["vectors"].shape == (rows * cols, cfg.COLPALI_DIM)
        assert torch.allclose(p["vectors"].float().norm(dim=1), torch.ones(rows * cols), atol=0.02)
