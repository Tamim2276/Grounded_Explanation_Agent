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
    if cfg.BACKEND == "real":
        return real_text_retriever(corpus, query, section, k)
    k = k or cfg.TOP_K
    pool = [c for c in corpus["chunks"] if c["section"] == section] or corpus["chunks"]
    ranked = sorted(pool, key=lambda c: -score(query, c["text"], corpus))
    return [Evidence("text", c["id"], c["page"], c["text"], c["bbox"], "TextRetriever",
                     score(query, c["text"], corpus), query=query, section=c["section"])
            for c in ranked[:k] if score(query, c["text"], corpus) > 0]


def in_section(chunk: dict, section: str) -> bool:
    # "Section 4" means section 4 and its subsections 4.1, 4.2, ...
    return chunk["section"] == section or chunk["section"].startswith(section + ".")


def ranks(scores: dict) -> dict:
    # 1 = best; ties share the better rank, so equal scores never get an arbitrary order.
    values = list(scores.values())
    return {key: 1 + sum(v > s for v in values) for key, s in scores.items()}


def text_scores(meaning: dict, words: dict, how: str) -> dict:
    # The score each paragraph is ranked by. "hybrid" is reciprocal rank fusion: a
    # paragraph earns 1 / (RRF_K + its rank) from each search, so one that BOTH searches
    # rank high wins, and the two scales (cosine about 0.7, keyword 0-10) never have to
    # be made comparable. A search that does not find an item at all (score 0: no
    # shared word) gives it nothing -- otherwise every non-match would tie for a rank
    # just behind the one real match and earn almost as much.
    if how == "specter2":
        return dict(meaning)
    if how == "keyword":
        return dict(words)
    if how == "hybrid":
        by_meaning, by_words = ranks(meaning), ranks(words)
        return {key: round((1 / (cfg.RRF_K + by_meaning[key]) if meaning[key] > 0 else 0.0)
                           + (1 / (cfg.RRF_K + by_words[key]) if words[key] > 0 else 0.0), 5)
                for key in meaning}
    raise ValueError(f"unknown TEXT_SEARCH {how!r}")


def meaning_scores(corpus: dict, query: str) -> dict:
    # SPECTER2 similarity of the query (question adapter, CPU) to every row of the
    # paper's text index -- paragraphs and captions. Empty for keyword search, which
    # needs no model and no index.
    if cfg.TEXT_SEARCH == "keyword":
        return {}
    from gea.indexes import embed_text
    index = corpus["text_index"]
    if "faiss" not in index:
        raise FileNotFoundError(f"{corpus['paper_id']}: no text index -- run: python scripts/build_indexes.py text")
    sims, rows = index["faiss"].search(embed_text([query], kind="query"), index["faiss"].ntotal)
    return {index["ids"][r]: float(s) for s, r in zip(sims[0], rows[0])}


def real_text_retriever(corpus: dict, query: str, section: Optional[str] = None, k: int = None) -> list:
    # N6a on a real paper (BUILD_PLAN.md Day 3): the query's SPECTER2 vector (question
    # adapter, on the CPU) against the paper's FAISS index, and/or the keyword score,
    # as cfg.TEXT_SEARCH says. Only paragraphs come back -- caption rows are for the
    # table and figure tools -- inside N2's section when one was named and matches.
    k = k or cfg.TEXT_TOP_K
    pool = [c for c in corpus["chunks"] if section and in_section(c, section)] or corpus["chunks"]
    words = {c["id"]: score(query, c["text"], corpus) for c in pool}
    similarity = meaning_scores(corpus, query)
    meaning = {c["id"]: similarity.get(c["id"], 0.0) for c in pool}
    final = text_scores(meaning, words, cfg.TEXT_SEARCH)
    ranked = sorted(pool, key=lambda c: (-final[c["id"]], -meaning[c["id"]]))
    return [Evidence("text", c["id"], c["page"], c["text"], tuple(c["bbox"]), "TextRetriever",
                     round(final[c["id"]], 4), query=query, section=c["section"])
            for c in ranked[:k]]


def n6a_text_retriever(state: AgentState) -> dict:
    # N6a -- TextRetriever: narrative chunks from the text index.
    # With cfg.BACKEND = "real", text_retriever runs real_text_retriever (Day 3);
    # this node is the same for both.
    action = state.get("action")
    assert action and action["tool"] == "text", "N6a: the planner did not ask for text"
    corpus = get_corpus(state["paper_id"])
    found = text_retriever(corpus, action["query"], section=state["query_filters"]["refs"].get("section"))
    return {"new_evidence": found,
            "events": step("N6a", describe_found(found), short="TextRetriever")}


# --- tables and figures on the real backend (N6b, N6c; BUILD_PLAN.md Day 4) ----------
# Every table (or figure) of the paper is scored two ways, and the two are combined by
# rank, as in N6a's hybrid search:
#   text   -- N6a's ranking on its words: SPECTER2 on the caption, keyword on the caption
#             plus the words printed inside its region (a table's cells, a chart's labels)
#   visual -- how hot ColQwen2's heat map is inside its region on its page
# cfg.REGION_SEARCH picks "text", "visual" or "both". Without ColQwen2 (no room on this
# PC and no cached query), "visual" is unavailable and the text decides alone.

def page_size(corpus: dict, page: int) -> tuple:
    return tuple(corpus["pages"][page - 1]["size"])


def region_words(corpus: dict, record: dict) -> str:
    # A table's or figure's caption plus the words printed inside its region on the page,
    # read from the PDF once per paper. A question about "4-hop triples" matches a table
    # whose CELLS say so, even when its caption only says "Basic statistics".
    cache = corpus.setdefault("_region_words", {})
    if not cache:
        import pymupdf
        with pymupdf.open(corpus["pdf"]) as doc:
            for r in corpus["tables"] + corpus["figures"]:
                cache[r["id"]] = " ".join(doc[r["page"] - 1].get_textbox(pymupdf.Rect(r["bbox"])).split())
    return f"{record['caption']} {cache.get(record['id'], '')}"


def page_heat(corpus: dict, query: str):
    # {page: heat map} for this query, or None when ColQwen2 cannot run now.
    from gea.indexes import page_query_vectors, similarity_map
    pages = corpus.get("page_index")
    q = page_query_vectors(query) if pages else None
    if q is None:
        return None
    return {p["page"]: similarity_map(q, p) for p in pages}


def inside_mask(heat: np.ndarray, bbox: tuple, size: tuple) -> np.ndarray:
    # Which patches have their centre inside the box (at least the one under its centre).
    rows, cols = heat.shape
    ys = (np.arange(rows) + 0.5) * size[1] / rows
    xs = (np.arange(cols) + 0.5) * size[0] / cols
    mask = (ys[:, None] >= bbox[1]) & (ys[:, None] <= bbox[3]) & (xs[None, :] >= bbox[0]) & (xs[None, :] <= bbox[2])
    if not mask.any():
        r = min(rows - 1, int((bbox[1] + bbox[3]) / 2 / size[1] * rows))
        c = min(cols - 1, int((bbox[0] + bbox[2]) / 2 / size[0] * cols))
        mask[r, c] = True
    return mask


def region_heat(heat: np.ndarray, bbox: tuple, size: tuple, top: int = 5) -> float:
    # How hot a region is: the mean of its `top` hottest patches.
    values = np.sort(heat[inside_mask(heat, bbox, size)])
    return float(values[-top:].mean())


def masked_heat(heat: np.ndarray, bbox: tuple, size: tuple) -> np.ndarray:
    # The map kept inside the region only, so N10b's box falls on the chosen table/figure.
    return np.where(inside_mask(heat, bbox, size), heat, heat.min())


def rank_regions(corpus: dict, query: str, kind: str, how: str = None) -> tuple:
    # All tables (kind="table") or figures of the paper, best first, as (record, score),
    # plus the heat maps used ({page: map}, or None) and the method actually applied.
    how = how or cfg.REGION_SEARCH
    records = corpus[kind + "s"]
    if not records:
        return [], None, how
    heat = page_heat(corpus, query) if how in ("visual", "both") else None   # None: no ColQwen2
    if heat is None and how != "text":
        how = "text"                                          # no ColQwen2 now: the words decide
    similarity = meaning_scores(corpus, query)
    words = text_scores({r["id"]: similarity.get(r["id"], 0.0) for r in records},
                        {r["id"]: score(query, region_words(corpus, r), corpus) for r in records},
                        cfg.TEXT_SEARCH)
    visual = ({r["id"]: region_heat(heat[r["page"]], r["bbox"], page_size(corpus, r["page"])) for r in records}
              if heat else {})
    final = {"text": words, "visual": visual,
             "both": text_scores(visual, words, "hybrid") if visual else words}[how]
    ranked = sorted(records, key=lambda r: (-final[r["id"]], -words[r["id"]]))
    return [(r, final[r["id"]]) for r in ranked], heat, how


def region_evidence(corpus: dict, record: dict, kind: str, s: float, query: str, heat) -> Evidence:
    # One table or figure as evidence: its region, its page size, the heat map kept to
    # the region (figures and tables alike, when ColQwen2 ran), and for a table every
    # cell with its box (parsed by scripts/build_indexes.py tables).
    size = page_size(corpus, record["page"])
    similarity = masked_heat(heat[record["page"]], record["bbox"], size) if heat else None
    common = dict(query=query, label=record["label"], similarity=similarity, page_size=size)
    if kind == "figure":
        return Evidence("figure", record["id"], record["page"], record["caption"], tuple(record["bbox"]),
                        "FigureRetriever", round(s, 4), **common)
    parsed = corpus.get("parsed_tables", {}).get(record["id"], {})
    if parsed.get("header"):
        table = {**record, **parsed}
        cells = [{"row": row[0], "col": table["header"][c], "value": value, "bbox": table["cell_boxes"][r + 1][c]}
                 for r, row in enumerate(table["rows"]) for c, value in enumerate(row)
                 if c < len(table["header"]) and table["cell_boxes"][r + 1][c]]
        content = table_text(table)
    else:                                                     # no grid: the region's raw text
        cells, content = [], "\n".join(x for x in (record["caption"], parsed.get("text", "")) if x)
    return Evidence("table", record["id"], record["page"], content, tuple(record["bbox"]),
                    "TableRetriever", round(s, 4), cells=cells, **common)


def real_region_retriever(corpus: dict, query: str, kind: str, ref: Optional[str] = None, k: int = None) -> list:
    # Real N6b / N6c: the table or figure the question names ("Table 2"), else the best
    # ranked ones.
    k = k or cfg.TOP_K
    label = f"{kind.capitalize()} {ref}" if ref else None
    named = [r for r in corpus[kind + "s"] if label and r["label"] == label]
    ranked, heat, _ = rank_regions(corpus, query, kind)
    chosen = [(r, 1.0) for r in named] or ranked[:k]
    return [region_evidence(corpus, r, kind, s, query, heat) for r, s in chosen]


# --- N6b: tables -------------------------------------------------------------------

def table_retriever(corpus: dict, query: str, k: int = None, ref: Optional[str] = None) -> list:
    # Rank tables by caption + cell text; return each with every cell's box.
    if cfg.BACKEND == "real":
        return real_region_retriever(corpus, query, "table", ref, k)
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
    # Real version (cfg.BACKEND = "real", Day 4): the table is chosen by caption and
    # ColQwen2's heat map; its cells come from the Table Transformer, each read from the
    # PDF at its position. Keeping the cell boxes is what makes N10b possible.
    action = state.get("action")
    assert action and action["tool"] == "table", "N6b: the planner did not ask for a table"
    found = table_retriever(get_corpus(state["paper_id"]), action["query"],
                            ref=state["query_filters"]["refs"].get("table"))
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
    if cfg.BACKEND == "real":
        return real_region_retriever(corpus, query, "figure", ref, k)
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
    # Real version (cfg.BACKEND = "real", Day 4): ColQwen2's heat map of each page
    # (no OCR) and the captions choose the figure. It keeps the heat map, because
    # N10b turns that map into a bounding box.
    action = state.get("action")
    assert action and action["tool"] == "figure", "N6c: the planner did not ask for a figure"
    found = figure_retriever(get_corpus(state["paper_id"]), action["query"],
                             ref=state["query_filters"]["refs"].get("figure"))
    return {"new_evidence": found,
            "events": step("N6c", describe_found(found), short="FigureRetriever")}


# --- N7: zoom ----------------------------------------------------------------------

def zoom_crop(corpus: dict, ev: Evidence) -> tuple:
    # Real N7: re-render the figure's region from the PDF at cfg.ZOOM_DPI (300), but no
    # bigger than cfg.ZOOM_MAX_PX on its long side (an image costs the model tokens).
    # Returns (PNG bytes, (height, width, 3)).
    import pymupdf
    with pymupdf.open(corpus["pdf"]) as doc:
        page = doc[ev.page - 1]
        rect = pymupdf.Rect(ev.bbox) & page.rect
        dpi = int(min(cfg.ZOOM_DPI, cfg.ZOOM_MAX_PX * 72 / max(rect.width, rect.height)))
        pix = page.get_pixmap(dpi=dpi, clip=rect)
        return pix.tobytes("png"), (pix.height, pix.width, 3)


def n7_region_zoom(state: AgentState) -> dict:
    # N7 -- Region Zoom. Crop the figure and re-render it at higher resolution.
    #
    # A page-level match says which page holds the chart, not what its curve does;
    # the zoomed view is what the generator reads. Real version (cfg.BACKEND = "real",
    # Day 4): PyMuPDF re-renders the region at 300 dpi -> PNG bytes in Evidence.crop.
    figures = [e for e in state.get("new_evidence", []) if e.kind == "figure"]
    assert figures, "N7: N6c returned no figure to zoom into"
    zoomed, shapes = [], []
    for f in figures:
        if cfg.BACKEND == "real":
            crop, shape = zoom_crop(get_corpus(state["paper_id"]), f)
        else:
            shape = (round((f.bbox[3] - f.bbox[1]) / 72 * 300), round((f.bbox[2] - f.bbox[0]) / 72 * 300), 3)
            crop = FakeTensor(shape, "300 dpi crop")
        shapes.append(shape)
        zoomed.append(Evidence("figure", f.source, f.page, f"Zoomed view (300 dpi) of: {f.content}",
                               f.bbox, "RegionZoom", f.score, query=f.query, label=f.label,
                               crop=crop, page_size=f.page_size))
    return {"new_evidence": state["new_evidence"] + zoomed,
            "events": step("N7", f"re-rendered {len(zoomed)} region(s) at 300 dpi -> "
                                 f"{shapes[0]}", short="RegionZoom")}


def route_zoom(state: AgentState) -> str:
    # Conditional edge out of N6c: zoom when the figure is small on its page (dense
    # detail), otherwise go straight to the buffer. Returns a ROUTE NAME only.
    figures = [e for e in state.get("new_evidence", []) if e.kind == "figure"]
    if not figures:
        return "buffer"
    x0, y0, x1, y1 = figures[0].bbox
    page_w, page_h = figures[0].page_size or (cfg.PAGE_W, cfg.PAGE_H)     # real papers: often US Letter
    return "zoom" if (x1 - x0) * (y1 - y0) / (page_w * page_h) < cfg.ZOOM_IF_AREA_BELOW else "buffer"


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
