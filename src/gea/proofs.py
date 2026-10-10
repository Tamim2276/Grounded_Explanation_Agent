# N10a-N10c, the three proofs, and N11-N12, the report and the answer.
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
    # The part of `box` inside `frame` (patches are 25 points wide; the edge ones stick out).
    return (max(box[0], frame[0]), max(box[1], frame[1]), min(box[2], frame[2]), min(box[3], frame[3]))


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
                     else f"cell {cite['cell'][0]} / {cite['cell'][1]}")
            sources.append({"evidence": ev.id, "kind": ev.kind, "page": ev.page, "where": where})
        links.append({"claim": i, "sources": sources})
    n_sources = sum(len(link["sources"]) for link in links)
    return {"claim_links": links,
            "events": step("N10a", f"{len(links)} claim(s) linked to {n_sources} source(s)")}


def n10b_bounding_boxes(state: AgentState) -> dict:
    # N10b -- boxes [x_min, y_min, x_max, y_max] on the page, in PDF points.
    # Table claims: the exact cells they cite (N6b kept every cell's box). Figure
    # claims: the box around the best patches of N6c's similarity map. Text claims:
    # the paragraph (notebook Section 13 narrows it to the quoted sentence).
    evidence = {e.id: e for e in state["evidence"]}
    boxes = []
    for i, claim in enumerate(state["claims"], 1):
        if claim.get("rejected"):
            continue
        for cite in claim["cites"]:
            ev = evidence[cite["evidence"]]
            if "cell" in cite:
                cell = next(c for c in ev.cells if (c["row"], c["col"]) == tuple(cite["cell"]))
                box, what = cell["bbox"], f"{ev.label} cell {cite['cell'][0]} / {cite['cell'][1]}"
            elif ev.kind == "figure" and ev.similarity is not None:
                box = patches_to_bbox(ev.similarity, page_size=ev.page_size)
                if ev.page_size:                # real map, kept to the figure: clip the patch edges to it
                    box = clip(box, ev.bbox)
                what = f"{ev.label} (patch map)"
            elif ev.tool == "RegionZoom":
                continue                    # a zoomed view points at the same region as its figure
            elif ev.kind == "figure":
                box, what = ev.bbox, f"{ev.label} (region)"     # no heat map (ColQwen2 did not run)
            elif ev.kind == "table":
                box, what = ev.bbox, f"{ev.label} (region)"      # a quote of the table, not a cell
            else:
                box, what = ev.bbox, f"Section {ev.section} paragraph"
            boxes.append({"claim": i, "evidence": ev.id, "page": ev.page,
                          "bbox": tuple(round(v, 1) for v in box), "what": what})
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
    return {"answer": answer,
            "events": step("N12", f"status={state.get('status', 'ok')}: {answer[:64]}...")}
