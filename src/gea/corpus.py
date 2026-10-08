# The offline path (N3 -> N4) and get_corpus(), which every retriever searches.
import json
import math
from collections import Counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from gea import config as cfg
from gea.state import FakeTensor
from gea.stub_paper import FAKE_PAPER
from gea.text import table_text, terms
from gea.trace import trace

PAPERS = {FAKE_PAPER["paper_id"]: FAKE_PAPER}      # the stub corpus: one paper


def grid_boxes(table: dict) -> list:
    # Page coordinates of every cell, row by row, header row first.
    # Real version: the Table Transformer returns these boxes. Here they follow from
    # the table's ruled grid -- column widths and row height -- which is exactly how
    # the table is drawn in notebook Section 13.
    x0, y0 = table["bbox"][:2]
    xs = [x0]
    for w in table["col_widths"]:
        xs.append(xs[-1] + w)
    n_rows = 1 + len(table["rows"])
    return [[(xs[c], y0 + r * table["row_h"], xs[c + 1], y0 + (r + 1) * table["row_h"])
             for c in range(len(xs) - 1)]
            for r in range(n_rows)]


class IngestState(TypedDict, total=False):
    # State of the offline graph. Runs once per paper, before any question.
    paper_id:   str
    pages:      list      # N3 -> one record per rendered page
    chunks:     list      # N3 -> narrative text with section, page and bbox
    tables:     list      # N3 -> tables with header, rows and a box per cell
    figures:    list      # N3 -> figures with caption and bbox
    text_index: dict      # N4 -> SPECTER2 vectors (+ the keyword stand-in)
    page_index: dict      # N4 -> ColQwen2 patch embeddings per page


def n3_corpus_ingestion(state: IngestState) -> dict:
    # N3 -- Corpus Ingestion. Every page becomes an image; its text, tables and
    # figures become records that keep their page coordinates.
    #
    # Real version (BUILD_PLAN.md Day 2): PyMuPDF renders each page and extracts
    # the text blocks, captions, tables and figures with their boxes.
    paper = PAPERS.get(state.get("paper_id"))
    assert paper, f"N3: unknown paper {state.get('paper_id')!r}"
    pages = [{"page": p, "size": (cfg.PAGE_W, cfg.PAGE_H),
              "image": FakeTensor((1754, 1240, 3), "150 dpi render")}
             for p in range(1, paper["pages"] + 1)]
    tables = [{**t, "cell_boxes": grid_boxes(t)} for t in paper["tables"]]
    trace("N3", f"{len(pages)} pages -> {len(paper['chunks'])} chunks, "
                f"{len(tables)} table(s), {len(paper['figures'])} figure(s)")
    return {"pages": pages, "chunks": paper["chunks"], "tables": tables,
            "figures": paper["figures"]}


def n4_dual_index_build(state: IngestState) -> dict:
    # N4 -- Dual Index Build: one index over the text, one over the page images.
    #
    # Real version (BUILD_PLAN.md Day 3): SPECTER2 embeds every chunk and caption
    # into FAISS; ColQwen2 embeds every page image as patches of COLPALI_DIM. The
    # stub keeps those shapes, and an IDF table over the same texts stands in for
    # similarity, so retrieval is deterministic.
    chunks, tables, figures, pages = (state.get(k) for k in ("chunks", "tables", "figures", "pages"))
    assert chunks and pages, "N4: N3 did not run (no chunks or pages in state)"
    docs = ([c["text"] for c in chunks] + [table_text(t) for t in tables]
            + [f["caption"] for f in figures])
    df = Counter(t for d in docs for t in set(terms(d)))
    idf = {t: math.log(1 + len(docs) / (1 + n)) for t, n in df.items()}
    n_patches = cfg.PATCH_GRID[0] * cfg.PATCH_GRID[1]
    text_index = {"vectors": FakeTensor((len(docs), cfg.SPECTER2_DIM), "SPECTER2"), "idf": idf}
    page_index = {"patches": FakeTensor((len(pages), n_patches, cfg.COLPALI_DIM), "ColQwen2"),
                  "grid": cfg.PATCH_GRID}
    trace("N4", f"text index {text_index['vectors'].shape}, page index {page_index['patches'].shape}")
    return {"text_index": text_index, "page_index": page_index}


def build_ingest_graph():
    # The offline path: N3 -> N4. Runs once per corpus, before any question.
    g = StateGraph(IngestState)
    g.add_node("n3_corpus_ingestion", n3_corpus_ingestion)
    g.add_node("n4_dual_index_build", n4_dual_index_build)
    g.add_edge(START, "n3_corpus_ingestion")
    g.add_edge("n3_corpus_ingestion", "n4_dual_index_build")
    g.add_edge("n4_dual_index_build", END)
    return g.compile()


_CORPORA = {}


def keyword_index(docs: list) -> dict:
    # IDF of every word over the paper's texts: the stand-in similarity behind
    # text.score(). The stub G1 and N9 still use it until Days 5-6.
    df = Counter(t for d in docs for t in set(terms(d)))
    return {t: math.log(1 + len(docs) / (1 + n)) for t, n in df.items()}


def load_real_corpus(paper_id: str) -> dict:
    # A real paper as scripts/build_corpus.py saved it (N3, Day 2), plus its keyword
    # index. Day 3 adds the SPECTER2 and ColQwen2 indexes here.
    path = cfg.DATA_DIR / "corpus" / paper_id / "corpus.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run: python scripts/build_corpus.py")
    paper = json.loads(path.read_text(encoding="utf-8"))
    docs = ([c["text"] for c in paper["chunks"]]
            + [r["caption"] for r in paper["tables"] + paper["figures"]])
    paper["text_index"] = {"idf": keyword_index(docs)}
    return paper


def get_corpus(paper_id: str) -> dict:
    # Everything one paper offers the retrievers: N3's records plus N4's two
    # indexes. Built once per paper and kept for the rest of the session.
    #
    # Stub backend: run the offline graph on the fake paper.
    # Real backend: load what scripts/build_corpus.py saved under data/corpus/.
    key = (cfg.BACKEND, paper_id)
    if key not in _CORPORA:
        if cfg.BACKEND == "stub":
            _CORPORA[key] = build_ingest_graph().invoke({"paper_id": paper_id})
        elif cfg.BACKEND == "real":
            _CORPORA[key] = load_real_corpus(paper_id)
        else:
            raise ValueError(f"unknown BACKEND {cfg.BACKEND!r}")
    return _CORPORA[key]
