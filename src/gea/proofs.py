# N10a-N10c, the three proofs, and N11-N12, the report and the answer.
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from gea import config as cfg
from gea.state import AgentState
from gea.trace import step


def patches_to_bbox(sim: np.ndarray, keep: float = 0.5, page_size: tuple = None) -> tuple:
    # Turn a patch similarity map into one box in PDF points: keep every patch at least
    # `keep` of the way from the coolest to the hottest, take the smallest rectangle
    # around them, and scale patch indices to the page (x = column x width / columns).
    # Real maps sit in a narrow band (0.2-0.5), so the band is stretched to 0-1 first.
    rows, cols = sim.shape
    span = sim.max() - sim.min()
    ys, xs = np.nonzero((sim - sim.min()) >= keep * span) if span > 0 else np.nonzero(sim == sim.max())
    page_w, page_h = page_size or (cfg.PAGE_W, cfg.PAGE_H)
    ph, pw = page_h / rows, page_w / cols
    return (round(float(xs.min() * pw), 1), round(float(ys.min() * ph), 1),
            round(float((xs.max() + 1) * pw), 1), round(float((ys.max() + 1) * ph), 1))


def clip(box: tuple, frame: tuple) -> tuple:
    # The part of `box` inside `frame` (patches are 25 points wide; the edge ones stick out),
    # or the frame itself when the box lies wholly outside it.
    inner = (max(box[0], frame[0]), max(box[1], frame[1]), min(box[2], frame[2]), min(box[3], frame[3]))
    return inner if inner[0] < inner[2] and inner[1] < inner[3] else tuple(frame)


def iou(a: tuple, b: tuple) -> float:
    # Intersection over union of two boxes -- the CGS metric of the thesis.
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def n10a_claim_links(state: AgentState) -> dict:
    # N10a -- claim -> source links: for every published claim, the evidence it
    # rests on and where in that evidence (the quote or the cell).
    evidence = {e.id: e for e in state["evidence"]}
    links = []
    for i, claim in enumerate(state["claims"], 1):
        if claim.get("rejected"):
            continue
        sources = []
        for cite in claim["cites"]:
            ev = evidence[cite["evidence"]]
            where = (f"\"{cite['quote'][:40]}...\"" if "quote" in cite
                     else f"cell {cite['cell'][0]} / {cite['cell'][1]}" if "cell" in cite
                     else f"observation \"{cite['observation'][:40]}...\"")      # a figure, as seen (Day 6)
            sources.append({"evidence": ev.id, "kind": ev.kind, "page": ev.page, "where": where})
        links.append({"claim": i, "sources": sources})
    n_sources = sum(len(link["sources"]) for link in links)
    return {"claim_links": links,
            "events": step("N10a", f"{len(links)} claim(s) linked to {n_sources} source(s)")}


def quote_box(page, quote: str, frame: tuple = None):
    # The box around a quote as printed on the page, or None: its first occurrence, joined
    # with the next lines when the quote wraps (a label can occur several times in a
    # diagram). If the whole quote is not found (a word split at a line end), its first
    # eight words are tried.
    import pymupdf
    clip = pymupdf.Rect(frame) if frame else None
    words = quote.split()
    for text in (quote, " ".join(words[:8]) if len(words) > 8 else None):
        hits = sorted(page.search_for(text, clip=clip) if text else [], key=lambda r: (r.y0, r.x0))
        if hits:
            group = [hits[0]]
            for r in hits[1:]:
                if abs(r.y0 - group[-1].y0) < 1 and 0 <= r.x0 - group[-1].x1 < 12:
                    group.append(r)                      # the same line, split at a wide justified gap
                elif 0 <= r.y0 - group[-1].y1 < 3:       # the quote continues on the next line
                    group.append(r)
                elif r.y0 > group[-1].y1:
                    break
            return (min(r.x0 for r in group), min(r.y0 for r in group), max(r.x1 for r in group), max(r.y1 for r in group))
    return None


FILLER = frozenset("""a an and are as at be by for from in into is it its of on or that the this these those to
with which while both each two three their there than between figure fig show shows shown showing image picture
view zoomed diagram chart plot graph panel left right top bottom part""".split())


def label_words(text: str) -> list:
    # The words that can name something printed in a figure: ligatures and hyphens undone,
    # lower case, a plural "s" dropped, at least three characters with a letter (no axis
    # ticks), no filler.
    words = unicodedata.normalize("NFKC", text).lower().replace("-", " ").split()
    words = [w.strip(".,:;!?()[]{}\"'`") for w in words]
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]
    return [w for w in words if len(w) >= 3 and any(ch.isalpha() for ch in w) and w not in FILLER]


def label_box(page, text: str, frame: tuple):
    # The box around the labels printed in a figure that an observation names, and the
    # labels, or (None, []). A label is one printed line inside the figure; it is named
    # when ALL its words occur in the observation -- "two auxiliary tasks: pCTR and
    # pCTCVR" names "Auxiliary Tasks", "pCTR" and "pCTCVR", not "Main Task". A photo has
    # no printed words and gets None.
    import pymupdf
    named, lines = set(label_words(text)), defaultdict(list)
    for w in page.get_text("words", clip=pymupdf.Rect(frame)):
        lines[(w[5], w[6])].append(w)                   # (block, line): one printed label
    found = [ws for ws in lines.values()
             if (need := label_words(" ".join(w[4] for w in ws))) and all(x in named for x in need)]
    if not found:
        return None, []
    words = [w for ws in found for w in ws]
    box = (min(w[0] for w in words), min(w[1] for w in words), max(w[2] for w in words), max(w[3] for w in words))
    return box, list(dict.fromkeys(" ".join(w[4] for w in ws) for ws in found))


def exact_box(page, ev, cite: dict, record: dict = None):
    # Real pages: where a quote or an observation is printed, as (box, what), or None.
    # A text quote: the sentence. A figure quote: a label inside the figure, else the
    # caption (it sits outside the figure's box). A figure observation: the labels it names.
    if "quote" in cite and ev.kind == "text":
        box = quote_box(page, cite["quote"])
        return (box, f"Section {ev.section} sentence") if box else None
    if ev.kind != "figure":
        return None
    if "quote" in cite:
        box = quote_box(page, cite["quote"], ev.bbox)
        if box:
            return box, f"{ev.label} label"
        box = quote_box(page, cite["quote"], tuple(record["caption_bbox"])) if record and record.get("caption_bbox") else None
        return (box, f"{ev.label} caption") if box else None
    if "observation" in cite:
        box, labels = label_box(page, cite["observation"], ev.bbox)
        return (box, f"{ev.label} labels: " + ", ".join(labels)[:80]) if box else None
    return None


def n10b_bounding_boxes(state: AgentState) -> dict:
    # N10b -- boxes [x_min, y_min, x_max, y_max] on the page, in PDF points.
    # Table claims: the exact cells they cite (N6b kept every cell's box); a bar of a
    # chart read from the PDF is a cell too. Figure claims: the box around the best
    # patches of N6c's similarity map. Text claims: the paragraph (notebook Section 13
    # narrows it to the quoted sentence). On the real backend (Day 6) a quote is found on
    # the page -- the exact sentence, the exact label inside a diagram, or the caption --
    # and an observation boxes the diagram labels it names; the map is the fallback (a
    # photo has no printed words): the claim's OWN map when its ColQwen2 vectors are cached
    # (Day 7's second pass), else the question's. Every box also records its page size and
    # cite kind (the website draws them itself).
    evidence = {e.id: e for e in state["evidence"]}
    doc, records, corpus = None, {}, None
    if cfg.BACKEND == "real":
        import pymupdf
        from gea.corpus import get_corpus
        from gea.retrieval import claim_heat
        corpus = get_corpus(state["paper_id"])
        doc, records = pymupdf.open(corpus["pdf"]), {r["id"]: r for r in corpus["figures"]}
    boxes = []
    for i, claim in enumerate(state["claims"], 1):
        if claim.get("rejected"):
            continue
        cited = [evidence[x["evidence"]] for x in claim["cites"]]
        for cite in claim["cites"]:
            ev = evidence[cite["evidence"]]
            if ev.tool == "RegionZoom" and any(e.source == ev.source and e.tool != "RegionZoom" for e in cited):
                continue                    # a zoomed view points at the same region as its figure
            # a zoomed view cited alone is boxed like its figure, with the figure's map
            heat = ev.similarity if ev.tool != "RegionZoom" else next(
                (e.similarity for e in evidence.values() if e.source == ev.source and e.tool != "RegionZoom"), None)
            found = exact_box(doc[ev.page - 1], ev, cite, records.get(ev.source)) \
                if doc is not None and "cell" not in cite else None
            if "cell" in cite:
                cell = next((c for c in ev.cells if (c["row"], c["col"]) == tuple(cite["cell"])), None)
                if cell is None:
                    continue
                part = "bar" if ev.kind == "figure" else "cell"
                box, what = cell["bbox"], f"{ev.label} {part} {cite['cell'][0]} / {cite['cell'][1]}"
            elif found is not None:
                box, what = found
            elif ev.kind == "figure" and (own := claim_heat(corpus, claim["text"], ev) if doc is not None else None) is not None:
                box, what = clip(patches_to_bbox(own, page_size=ev.page_size), ev.bbox), f"{ev.label} (claim map)"
            elif ev.kind == "figure" and heat is not None:
                box = patches_to_bbox(heat, page_size=ev.page_size)
                if ev.page_size:                # real map, kept to the figure: clip the patch edges to it
                    box = clip(box, ev.bbox)
                what = f"{ev.label} (patch map)"
            elif ev.kind == "figure":
                box, what = ev.bbox, f"{ev.label} (region)"     # no heat map (ColQwen2 did not run)
            elif ev.kind == "table":
                box, what = ev.bbox, f"{ev.label} (region)"      # a quote of the table, not a cell
            else:
                box, what = ev.bbox, f"Section {ev.section} paragraph"
            record = {"claim": i, "evidence": ev.id, "page": ev.page,
                      "bbox": tuple(round(v, 1) for v in box), "what": what}
            if doc is not None:
                record.update(page_size=ev.page_size,
                              cite=next(k for k in ("cell", "quote", "observation") if k in cite))
            boxes.append(record)
    if doc is not None:
        doc.close()
    regions = len({(b["page"], b["bbox"]) for b in boxes})
    return {"boxes": boxes, "events": step("N10b", f"{len(boxes)} box(es) over {regions} region(s)")}


TOOL_SHORTS = ("TextRetriever", "TableRetriever", "FigureRetriever", "RegionZoom")


def n10c_dag_trace(state: AgentState) -> dict:
    # N10c -- the DAG trace: which tools ran, in what order, and every gate
    # decision, taken from the step log. The process can be checked, not trusted.
    steps = [e for e in state.get("events", []) if e["short"]]
    counts = Counter(e["short"] for e in steps)
    dag = {"path": " -> ".join(e["short"] for e in steps),
           "tool_calls": sum(counts[t] for t in TOOL_SHORTS),
           "g1_rejections": counts["G1 reject"],
           "g2_rejections": counts["G2 reject"],
           "steps": steps}
    return {"dag_trace": dag, "events": step("N10c", dag["path"])}


def n11_report_synthesis(state: AgentState) -> dict:
    # N11 -- Report Synthesis: the answer and its three proofs in one report.
    # What is published, what is withheld, and in what order are policy -- they
    # stay in code, not in a model.
    dag = state.get("dag_trace")
    assert dag, "N11: N10c has not run (no dag_trace in state)"
    evidence = state.get("evidence", [])
    lines = [
        f"GROUNDED ANSWER -- {state['paper_id']}",
        "=" * 72,
        f"Question  : {state['question']}",
        "Evidence  : " + (", ".join(f"{e.id} {e.kind} p.{e.page}" + (f" ({e.label})" if e.label else "")
                                     for e in evidence) or "none"),
        f"Retrieval : {state['round']} round(s), {dag['g1_rejections']} rejected by G1, "
        f"{dag['tool_calls']} tool call(s)",
        f"Trace     : {dag['path']}",
        "",
    ]
    if state.get("status") == "insufficient_evidence":
        lines += ["NO ANSWER -- the evidence is not sufficient", "-" * 72,
                  f"  still missing: {state['sufficiency']['missing']}", ""]
    else:
        boxes = defaultdict(list)
        for b in state.get("boxes", []):
            boxes[b["claim"]].append(f"p.{b['page']} {list(b['bbox'])} {b['what']}")
        kept = [(i, c) for i, c in enumerate(state["claims"], 1) if not c.get("rejected")]
        lines += [f"ANSWER ({len(kept)} claim(s))", "-" * 72]
        if state.get("best_effort"):
            lines.append("  (best effort: the search budget ran out before the evidence was judged complete)")
        for i, c in kept:
            ids = ", ".join(dict.fromkeys(x["evidence"] for x in c["cites"]))
            lines.append(f"{i}. {c['text']}  [{ids}]")
            lines += [f"     box {b}" for b in boxes.get(i, [])]
        withheld = [c for c in state["claims"] if c.get("rejected")]
        if withheld:
            lines += ["", "WITHHELD -- failed verification against the buffer", "-" * 72]
            lines += [f"  {c['text']}" for c in withheld]
        lines.append("")
    return {"report": "\n".join(lines),
            "events": step("N11", f"report written ({len(lines)} lines)")}


def n12_final_output(state: AgentState) -> dict:
    # N12 -- the terminal node: the short answer the user reads first.
    report = state.get("report")
    assert report, "N12: N11 produced no report"
    if state.get("status") == "insufficient_evidence":
        answer = ("I could not confirm an answer from this paper. Still missing: "
                  + state["sufficiency"]["missing"] + ".")
    else:
        kept = [c for c in state["claims"] if not c.get("rejected")]
        best = next((c for c in kept if c.get("conclusion")), kept[0] if kept else None)
        answer = best["text"] if best else "Every claim failed verification; nothing can be stated."
        if best and state.get("best_effort"):
            answer += " (The evidence found may be incomplete.)"
    return {"answer": answer,
            "events": step("N12", f"status={state.get('status', 'ok')}: {answer[:64]}...")}
