# Bar charts read from the PDF itself (BUILD_PLAN.md Day 6). Many CS papers embed plots as
# vector graphics: every bar is an exact rectangle and every axis number is PDF text. So
# instead of asking the 7B model to read bar heights off a picture -- it got 12 for a bar
# of ~55 -- each panel's y-axis is calibrated from its tick labels and each bar's top is
# converted to a value. The bars then behave like table cells (row = panel + series,
# column = x label), so the generator cites them like cells, G2 checks them exactly, and
# N10b's box is the bar itself.
#
# Handled: vertical bars on linear axes, grouped bars, several panels, axes that start
# above zero. Not handled (the figure then simply has no values): horizontal or stacked
# bars, log axes, line charts, raster images.
import json
import re
from pathlib import Path

import numpy as np
import pymupdf

from gea.safeio import atomic_write_json

NUM = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
PANEL = re.compile(r"^\([a-z]\)$")
CHARTS_DIR = "charts"        # data/corpus/<paper>/charts/<figure id>.json, one per figure


def words_in(page: pymupdf.Page, region: pymupdf.Rect) -> list:
    return [{"text": w[4], "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
             "cx": (w[0] + w[2]) / 2, "cy": (w[1] + w[3]) / 2}
            for w in page.get_text("words") if region.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))]


def y_axes(numbers: list) -> list:
    # Columns of tick labels: numbers sharing a right edge, read top to bottom with values
    # falling. A column ends where the values stop falling or a gap opens: two panels
    # stacked above each other share an x-position (Figure 4 of 1803.03467v4).
    axes = []
    for x1 in sorted({round(n["x1"]) for n in numbers}):
        stack = sorted([n for n in numbers if abs(n["x1"] - x1) < 2], key=lambda n: n["cy"])
        run = []
        for n in stack + [None]:
            if n is None or (run and (float(n["text"]) >= float(run[-1]["text"]) or n["cy"] - run[-1]["cy"] > 40)):
                if len(run) >= 3 and all(id(run[0]) != id(a[0]) for a in axes):
                    axes.append(run)
                run = []
            if n is not None:
                run.append(n)
    return axes


def calibrate(axis: list):
    # value = a * y + b from the tick labels; None if the ticks are not evenly spaced
    # (a log axis, or numbers that are not an axis).
    ys = np.array([t["cy"] for t in axis])
    vs = np.array([float(t["text"]) for t in axis])
    a, b = np.polyfit(ys, vs, 1)
    if np.abs(a * ys + b - vs).max() > 0.02 * (vs.max() - vs.min()):
        return None
    return a, b


def filled_rects(page: pymupdf.Page, region: pymupdf.Rect) -> list:
    return [(pymupdf.Rect(it[1]), tuple(round(c, 3) for c in d["fill"]))
            for d in page.get_drawings() if d.get("fill") and region.intersects(pymupdf.Rect(d["rect"]))
            for it in d["items"] if it[0] == "re"]


def legend_names(rects: list, words: list, bar_ids: set) -> dict:
    # colour -> series name: a small swatch (not a bar) with words to its right on its line.
    names = {}
    for i, (r, colour) in enumerate(rects):
        if i in bar_ids or not (2 < r.width < 30 and 2 < r.height < 15) or colour in names:
            continue
        line = sorted([w for w in words if abs(w["cy"] - (r.y0 + r.y1) / 2) < 2.5 and w["x0"] > r.x1 - 1],
                      key=lambda w: w["x0"])
        name, right = [], r.x1
        for w in line:
            if w["x0"] - right > 12:
                break
            name.append(w["text"])
            right = w["x1"]
        if name:
            names[colour] = " ".join(name)
    return names


def panel_name(words: list, x0: float, x1: float, below: float) -> str:
    # "(c) Bing-News": a "(c)" label under the panel, with the words after it on its line.
    marks = [w for w in words if PANEL.match(w["text"]) and w["cy"] > below and x0 - 10 <= w["cx"] <= x1 + 10]
    if not marks:
        return ""
    m = min(marks, key=lambda w: w["cy"])
    line = sorted([w for w in words if abs(w["cy"] - m["cy"]) < 4 and w["x0"] >= m["x0"]], key=lambda w: w["x0"])
    name = [line[0]["text"]]
    for w in line[1:]:
        if PANEL.match(w["text"]):                     # the next panel's label on the same line
            break
        name.append(w["text"])
    return " ".join(name)


def read_chart(page: pymupdf.Page, bbox: tuple):
    # {"kind": "bar", "bars": [{"panel", "series", "x", "value", "bbox"}]}, or None when the
    # region holds no readable bar chart (fewer than 3 bars).
    region = pymupdf.Rect(bbox)
    words = words_in(page, region)
    numbers = [w for w in words if NUM.match(w["text"])]
    rects = filled_rects(page, region)
    bars, bar_ids = [], set()
    for axis in y_axes(numbers):
        fit = calibrate(axis)
        if fit is None:
            continue
        a, b = fit
        top_y, floor_y = axis[0]["cy"], axis[-1]["cy"]
        zero_y = -b / a if a else None
        for i, (r, colour) in enumerate(rects):
            standing = abs(r.y1 - floor_y) < 3 or (zero_y is not None and abs(r.y1 - zero_y) < 3)
            if i in bar_ids or not standing or r.x0 < axis[0]["x1"] or r.height < 0.3 or not 1 < r.width < 40:
                continue
            if r.y0 < top_y - 15:                             # taller than the axis: a frame, not a bar
                continue
            nearer = [ax for ax in y_axes(numbers) if axis[0]["x1"] < ax[0]["x1"] < r.x0
                      and abs(ax[-1]["cy"] - floor_y) < 30]
            if nearer:                                        # another panel's axis sits in between
                continue
            bar_ids.add(i)
            # The visible floor is the lowest tick: plots whose axis starts above zero draw each
            # bar from 0 and clip it there (1706.04269v2 Fig. 4), so the rectangle reaches below.
            floor = floor_y if zero_y is None or zero_y > floor_y else zero_y
            bars.append({"rect": r, "colour": colour, "value": a * r.y0 + b, "axis": axis, "floor": floor})
    if len(bars) < 3:
        return None
    names = legend_names(rects, words, bar_ids)
    out = []
    for bar in bars:
        r, axis = bar["rect"], bar["axis"]
        under = [w for w in words if 2 < w["cy"] - bar["floor"] < 16 and w["x0"] > axis[0]["x1"]
                 and not PANEL.match(w["text"])]
        first = min((w["cy"] for w in under), default=None)       # the tick-label row, not the axis title below it
        row = [w for w in under if abs(w["cy"] - first) < 3]
        x = min(row, key=lambda w: abs(w["cx"] - (r.x0 + r.x1) / 2))["text"] if row else ""
        panel_bars = [b["rect"] for b in bars if b["axis"] is axis]
        panel = panel_name(words, axis[0]["x0"], max(p.x1 for p in panel_bars), bar["floor"] + 8)
        span = float(axis[0]["text"]) - float(axis[-1]["text"])
        value = round(bar["value"], 1 if span >= 10 else 2)
        out.append({"span": axis[-1]["cy"] - axis[0]["cy"], "raw": (r.width, r.height),
                    "panel": panel, "series": names.get(bar["colour"], f"colour {bar['colour']}"), "x": x,
                    "value": f"{value:g}",
                    "bbox": [round(v, 1) for v in (r.x0, r.y0, r.x1, min(r.y1, bar["floor"]))]})
    # Two safety rules: wrong values would be "verified" by G2 as exact.
    # 1. Line-chart markers resting near zero look like tiny bars: a series whose tallest
    #    "bar" is under 6% of its axis' height is dropped (1809.01246v1 Fig. 8 is a line chart).
    tallest = {}
    for b in out:
        key = (b["panel"], b["series"])
        tallest[key] = max(tallest.get(key, 0), (b["bbox"][3] - b["bbox"][1]) / b["span"])
    squares = {}                                       # line markers are small squares
    for b in out:
        w, h = b["raw"]                                # before clipping at the axis floor
        key = (b["panel"], b["series"])
        squares[key] = squares.get(key, True) and max(w, h) < 9 and abs(w - h) < 1.5
    out = [{k: v for k, v in b.items() if k not in ("span", "raw")} for b in out
           if tallest[(b["panel"], b["series"])] >= 0.06 and not squares[(b["panel"], b["series"])]]
    # 2. Values that cannot be named are not worth citing: most bars need an x label
    #    (rotated category labels, as in 1804.01429v3 Fig. 7, are not read).
    if len(out) < 3 or sum(bool(b["x"]) for b in out) < 0.8 * len(out):
        return None
    return {"kind": "bar", "bars": sorted(out, key=lambda b: (b["panel"], b["bbox"][0]))}


def chart_cells(chart: dict) -> list:
    # The bars as table cells: row = panel + series, column = x label.
    return [{"row": f"{b['panel']} {b['series']}".strip(), "col": b["x"], "value": b["value"], "bbox": tuple(b["bbox"])}
            for b in chart["bars"]]


def chart_text(chart: dict, limit: int = 1500) -> str:
    # The values as the generator reads them, one line per panel and series; cut at
    # `limit` characters (a histogram can have hundreds of bars -- all stay checkable).
    lines = {}
    for b in chart["bars"]:
        lines.setdefault(f"{b['panel']} {b['series']}".strip(), []).append(f"{b['x']}: {b['value']}")
    text = "Bar values read from the PDF: " + "; ".join(f"{k} -> {', '.join(v)}" for k, v in lines.items())
    return text if len(text) <= limit else text[:limit] + f" ... ({len(chart['bars'])} bars in all)"


def chart_file(folder: Path, figure_id: str) -> Path:
    return Path(folder) / CHARTS_DIR / f"{figure_id}.json"


def build_chart_cache(folder: Path, device: str = "cpu") -> int:
    # Read every figure of one paper once; one file per figure ({"kind": "none"} when it
    # holds no readable bar chart). Returns the number of figures read.
    folder = Path(folder)
    paper = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
    corpus_time = (folder / "corpus.json").stat().st_mtime
    todo = [f for f in paper["figures"] if not chart_file(folder, f["id"]).exists()
            or chart_file(folder, f["id"]).stat().st_mtime < corpus_time]
    if todo:
        with pymupdf.open(paper["pdf"]) as doc:
            for f in todo:
                chart = read_chart(doc[f["page"] - 1], tuple(f["bbox"])) or {"kind": "none"}
                atomic_write_json(chart_file(folder, f["id"]), {"id": f["id"], **chart})
    return len(todo)


def charts_are_current(folder: Path) -> bool:
    folder = Path(folder)
    paper = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
    corpus_time = (folder / "corpus.json").stat().st_mtime
    return all(chart_file(folder, f["id"]).exists() and chart_file(folder, f["id"]).stat().st_mtime >= corpus_time
               for f in paper["figures"])


def load_charts(folder: Path) -> dict:
    # figure id -> chart, for the figures with a readable bar chart
    found = {}
    for p in (Path(folder) / CHARTS_DIR).glob("*.json"):
        chart = json.loads(p.read_text(encoding="utf-8"))
        if chart.get("kind") == "bar":
            found[p.stem] = chart
    return found
