# N6a-N7, the retrieval tools, and N8, the evidence buffer.
#
# Only these nodes bring evidence in, and only N8 writes the buffer that N9 reads.
from typing import Optional

import numpy as np

from gea import config as cfg
from gea.corpus import get_corpus
from gea.state import AgentState, Evidence, FakeTensor
from gea.text import score, table_text
from gea.trace import step

TOOL_NODES = {"text": "n6a_text_retriever", "table": "n6b_table_retriever",
              "figure": "n6c_figure_retriever"}


def route_tool(state: AgentState) -> str:
    # Conditional edge out of N5: the tool the planner chose. Returns a ROUTE NAME only.
    return state["action"]["tool"]


def describe_found(found: list) -> str:
    if not found:
        return "found nothing"
    return "; ".join(f"{e.source} (page {e.page}, score {e.score}): "
                     f"\"{e.content.splitlines()[0][:48]}...\"" for e in found)


# --- N6a: text ---------------------------------------------------------------------

def text_retriever(corpus: dict, query: str, section: Optional[str] = None, k: int = None) -> list:
    # Rank narrative chunks against the query, inside one section if N2 found one.
    k = k or cfg.TOP_K
    pool = [c for c in corpus["chunks"] if c["section"] == section] or corpus["chunks"]
    ranked = sorted(pool, key=lambda c: -score(query, c["text"], corpus))
    return [Evidence("text", c["id"], c["page"], c["text"], c["bbox"], "TextRetriever",
                     score(query, c["text"], corpus), query=query, section=c["section"])
            for c in ranked[:k] if score(query, c["text"], corpus) > 0]


def n6a_text_retriever(state: AgentState) -> dict:
    # N6a -- TextRetriever: narrative chunks from the text index.
    # Real version (BUILD_PLAN.md Day 3): SPECTER2 query vector against FAISS.
    action = state.get("action")
    assert action and action["tool"] == "text", "N6a: the planner did not ask for text"
    corpus = get_corpus(state["paper_id"])
    found = text_retriever(corpus, action["query"], section=state["query_filters"]["refs"].get("section"))
    return {"new_evidence": found,
            "events": step("N6a", describe_found(found), short="TextRetriever")}


# --- N6b: tables -------------------------------------------------------------------

def table_retriever(corpus: dict, query: str, k: int = None) -> list:
    # Rank tables by caption + cell text; return each with every cell's box.
    k = k or cfg.TOP_K
    ranked = sorted(corpus["tables"], key=lambda t: -score(query, table_text(t), corpus))
    found = []
    for t in ranked[:k]:
        s = score(query, table_text(t), corpus)
        if s <= 0:
            continue
        cells = [{"row": row[0], "col": t["header"][c], "value": value,
                  "bbox": t["cell_boxes"][r + 1][c]}
                 for r, row in enumerate(t["rows"]) for c, value in enumerate(row)]
        found.append(Evidence("table", t["id"], t["page"], table_text(t), t["bbox"],
                              "TableRetriever", s, query=query, label=t["label"], cells=cells))
    return found


def n6b_table_retriever(state: AgentState) -> dict:
    # N6b -- TableRetriever: a table plus the page coordinates of every cell.
    #
    # Real version (BUILD_PLAN.md Day 4): ColQwen2 finds the page, the Table
    # Transformer recovers the grid, and each cell's text is read from the PDF at
    # that position. Keeping the cell boxes is what makes N10b possible.
    action = state.get("action")
    assert action and action["tool"] == "table", "N6b: the planner did not ask for a table"
    found = table_retriever(get_corpus(state["paper_id"]), action["query"])
    return {"new_evidence": found,
            "events": step("N6b", describe_found(found), short="TableRetriever")}


# --- N6c: figures ------------------------------------------------------------------

def fake_similarity_map(figure: dict, strength: float = 0.8) -> np.ndarray:
    # Stand-in for ColQwen2's per-patch similarity to the query, on PATCH_GRID.
    # Patches whose centre falls inside the figure score high, the rest low, plus
    # seeded noise. Real version: MaxSim of the query tokens against each page
    # patch, which colpali-engine returns as exactly this kind of grid.
    rows, cols = cfg.PATCH_GRID
    ys = (np.arange(rows) + 0.5) * cfg.PAGE_H / rows
    xs = (np.arange(cols) + 0.5) * cfg.PAGE_W / cols
    x0, y0, x1, y1 = figure["bbox"]
    inside = ((ys[:, None] >= y0) & (ys[:, None] <= y1)) & ((xs[None, :] >= x0) & (xs[None, :] <= x1))
    noise = np.random.default_rng(cfg.SEED).normal(0.0, 0.03, size=(rows, cols))
    return np.where(inside, strength, 0.1 * strength) + noise


def figure_retriever(corpus: dict, query: str, ref: Optional[str] = None, k: int = None) -> list:
    # Find figures: by number if the question names one, otherwise by caption.
    k = k or cfg.TOP_K
    if ref:
        ranked = [(1.0, f) for f in corpus["figures"] if f["label"] == f"Figure {ref}"]
    else:
        ranked = sorted(((score(query, f["caption"], corpus), f) for f in corpus["figures"]),
                        key=lambda sf: -sf[0])
    return [Evidence("figure", f["id"], f["page"], f["caption"], f["bbox"], "FigureRetriever",
                     s, query=query, label=f["label"], similarity=fake_similarity_map(f))
            for s, f in ranked[:k] if s > 0]


def n6c_figure_retriever(state: AgentState) -> dict:
    # N6c -- FigureRetriever (ColPali): charts and plots from the page index.
    #
    # Real version (BUILD_PLAN.md Day 4): ColQwen2 MaxSim against the page index,
    # no OCR. It keeps the per-patch similarity map, because N10b turns that map
    # into a bounding box.
    action = state.get("action")
    assert action and action["tool"] == "figure", "N6c: the planner did not ask for a figure"
    found = figure_retriever(get_corpus(state["paper_id"]), action["query"],
                             ref=state["query_filters"]["refs"].get("figure"))
    return {"new_evidence": found,
            "events": step("N6c", describe_found(found), short="FigureRetriever")}


# --- N7: zoom ----------------------------------------------------------------------

def n7_region_zoom(state: AgentState) -> dict:
    # N7 -- Region Zoom. Crop the figure and re-render it at higher resolution.
    #
    # A page-level match says which page holds the chart, not what its curve does;
    # the zoomed view is what the generator reads. Real version (BUILD_PLAN.md
    # Day 4): PyMuPDF re-renders the region at 300 dpi.
    figures = [e for e in state.get("new_evidence", []) if e.kind == "figure"]
    assert figures, "N7: N6c returned no figure to zoom into"
    zoomed = []
    for f in figures:
        w_px = round((f.bbox[2] - f.bbox[0]) / 72 * 300)
        h_px = round((f.bbox[3] - f.bbox[1]) / 72 * 300)
        zoomed.append(Evidence("figure", f.source, f.page, f"Zoomed view (300 dpi) of: {f.content}",
                               f.bbox, "RegionZoom", f.score, query=f.query, label=f.label,
                               crop=FakeTensor((h_px, w_px, 3), "300 dpi crop")))
    return {"new_evidence": state["new_evidence"] + zoomed,
            "events": step("N7", f"re-rendered {len(zoomed)} region(s) at 300 dpi -> "
                                 f"{zoomed[0].crop.shape}", short="RegionZoom")}


def route_zoom(state: AgentState) -> str:
    # Conditional edge out of N6c: zoom when the figure is small on its page (dense
    # detail), otherwise go straight to the buffer. Returns a ROUTE NAME only.
    figures = [e for e in state.get("new_evidence", []) if e.kind == "figure"]
    if not figures:
        return "buffer"
    x0, y0, x1, y1 = figures[0].bbox
    return "zoom" if (x1 - x0) * (y1 - y0) / (cfg.PAGE_W * cfg.PAGE_H) < cfg.ZOOM_IF_AREA_BELOW else "buffer"


# --- N8: the evidence buffer -------------------------------------------------------

def link_evidence(buffer: list) -> list:
    # Cross-modal links: same page, text that cites "Table N" / "Figure N", zoomed views.
    links = []
    for i, a in enumerate(buffer):
        for b in buffer[i + 1:]:
            why = ["same page"] if a.page == b.page else []
            for x, y in ((a, b), (b, a)):
                if x.kind == "text" and y.label and y.label in x.content:
                    why.append(f"text cites {y.label}")
            if a.source == b.source and "RegionZoom" in (a.tool, b.tool):
                why.append("zoomed view")
            if why:
                links.append((a.id, b.id, ", ".join(why)))
    return links


def n8_evidence_buffer(state: AgentState) -> dict:
    # N8 -- Cross-Modal Linking. The only way evidence reaches the generator.
    #
    # Gives every new item an id, adds it to the buffer (skipping anything already
    # there) and links related items. Nothing after this node looks at the corpus.
    new = state.get("new_evidence")
    assert new is not None, "N8: no retrieval tool ran (no new_evidence in state)"
    buffer = list(state.get("evidence", []))
    seen = {e.key for e in buffer}
    added = []
    for ev in new:
        if ev.key in seen:
            continue
        ev.id, ev.round = f"E{len(buffer) + 1}", state["round"]
        buffer.append(ev)
        seen.add(ev.key)
        added.append(ev)
    links = link_evidence(buffer)
    msg = ("added " + ", ".join(f"{e.id}={e.kind}/{e.source}" for e in added)) if added else "nothing new"
    msg += f"; buffer holds {len(buffer)} item(s)"
    if links:
        msg += "; links " + ", ".join(f"{a}-{b}" for a, b, _ in links)
    return {"evidence": buffer, "links": links, "new_evidence": [], "events": step("N8", msg)}
