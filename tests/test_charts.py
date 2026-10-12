# Bar charts read from the PDF (gea/charts.py, BUILD_PLAN.md Day 6), on real figures that
# were checked by eye. Skipped when the papers are not read (python scripts/build_corpus.py).
import json

import pymupdf
import pytest

from gea import config as cfg
from gea.charts import chart_cells, read_chart

CORPUS = cfg.DATA_DIR / "corpus"


def chart(pid: str, label: str):
    path = CORPUS / pid / "corpus.json"
    if not path.exists():
        pytest.skip(f"{pid} not read")
    paper = json.loads(path.read_text(encoding="utf-8"))
    f = next(r for r in paper["figures"] if r["label"] == label)
    with pymupdf.open(paper["pdf"]) as doc:
        return read_chart(doc[f["page"] - 1], tuple(f["bbox"])), f


def bar(found: dict, panel: str, series: str, x: str) -> dict:
    return next(b for b in found["bars"] if b["panel"].startswith(panel) and b["series"] == series and b["x"] == x)


def test_grouped_bars_in_four_panels():
    # The bar the 7B model read as 12 (true ~55), and the one it read as 25 (true ~4).
    found, _ = chart("1803.03467v4", "Figure 4")
    assert len(found["bars"]) == 24
    assert float(bar(found, "(c) Bing-News", "with common rater(s)", "4")["value"]) == pytest.approx(55.8, abs=1)
    assert float(bar(found, "(a) MovieLens-1M", "with common rater(s)", "1")["value"]) == pytest.approx(4.3, abs=0.5)


def test_an_axis_that_starts_above_zero():
    # Bars drawn from 0 and clipped at 75: the value is the top, the box stops at the axis.
    found, fig = chart("1706.04269v2", "Figure 4")
    b = bar(found, "", "Action Search", "2.5")
    assert float(b["value"]) == pytest.approx(103, abs=1.5)
    assert b["bbox"][3] <= fig["bbox"][3]                              # inside the figure
    assert {x["series"] for x in found["bars"]} == {"Action Search", "Direction Baseline", "Random Baseline"}


def test_bars_become_table_cells():
    found, _ = chart("1706.04269v2", "Figure 4")
    cells = {(c["row"], c["col"]): c["value"] for c in chart_cells(found)}
    assert cells[("Random Baseline", "2.5")] == "133.2"


@pytest.mark.parametrize("pid, label", [("1804.07931v2", "Figure 2"),      # an architecture diagram
                                        ("1809.01246v1", "Figure 8")])     # a line chart: markers are not bars
def test_no_values_where_there_is_no_bar_chart(pid, label):
    found, _ = chart(pid, label)
    assert found is None
