# N9, the grounded generator, and G2, the Faithfulness Gate, with the bounded
# regenerate loop between them.
#
# Every claim N9 writes carries `cites`: a word-for-word quote, or a table cell with
# its value. G2 never judges prose -- it runs equality tests against the buffer,
# which is the only thing N9 could read.
import re
from typing import Optional

from gea import config as cfg
from gea.corpus import get_corpus
from gea.state import AgentState, Evidence
from gea.text import NUMBER, score, sentences
from gea.trace import step, trace


# --- N9: the generator stand-in ----------------------------------------------------

def cite_cell(ev: Evidence, row: str, col: str) -> dict:
    # A citation of one table cell: which evidence, which cell, and its value.
    cell = next(c for c in ev.cells if c["row"] == row and c["col"] == col)
    return {"evidence": ev.id, "cell": (row, col), "value": cell["value"]}


def text_claims(ev: Evidence, corpus: dict) -> list:
    # Quote the sentence that best matches what the paragraph was retrieved for.
    best = max(sentences(ev.content), key=lambda s: score(ev.query, s, corpus))
    return [{"text": f'Section {ev.section} states: "{best}"',
             "cites": [{"evidence": ev.id, "quote": best}]}]


def table_claims(ev: Evidence, entities: list) -> list:
    # Report the row the question asks about, then compare it with the overall row.
    if not ev.cells:                                       # a table read without a grid
        first_line = ev.content.splitlines()[0]
        return [{"text": f"The most relevant table is {first_line}",
                 "cites": [{"evidence": ev.id, "quote": first_line}]}]
    rows = list(dict.fromkeys(c["row"] for c in ev.cells))
    header = [c["col"] for c in ev.cells if c["row"] == rows[0]]
    asked = [r for r in rows if r.lower() in {x.lower() for x in entities}]
    claims = []
    for row in asked:
        cites = [cite_cell(ev, row, col) for col in header[1:]]
        claims.append({"text": f"{ev.label} reports the {row.lower()} subset separately: "
                               + ", ".join(f"{c['cell'][1]} {c['value']}" for c in cites) + ".",
                       "cites": cites})
    overall = next((r for r in rows if r.lower() in ("all", "overall", "average")), None)
    if asked and overall and "Gain" in header:
        mine, total = cite_cell(ev, asked[0], "Gain"), cite_cell(ev, overall, "Gain")
        if cfg.TABLE_CLAIM_STYLE == "misattributed":      # notebook Section 15: another row's value
            other = next(r for r in rows if r not in (asked[0], overall))
            mine = cite_cell(ev, other, "Gain")
        holds = "does hold" if float(mine["value"]) > 0 else "does not hold"
        size = ("but it is smaller" if float(mine["value"]) < float(total["value"])
                else "and it is at least as large")
        text = (f"So the gain {holds} for the {asked[0].lower()} subset, {size}: "
                f"{mine['value']} points versus {total['value']} overall.")
        if cfg.TABLE_CLAIM_STYLE == "uncited":              # notebook Section 15: a number nobody cites
            text += " That is 61% of the high-resource gain."
        claims.append({"text": text, "cites": [mine, total], "conclusion": True})
    first_line = ev.content.splitlines()[0]
    return claims or [{"text": f"The most relevant table is {first_line}",
                       "cites": [{"evidence": ev.id, "quote": first_line}]}]


def figure_claims(ev: Evidence, evidence: list) -> list:
    # What the figure shows. A vision-language model reads the zoomed crop; the
    # stand-in states the caption, citing the figure and its zoomed view.
    zooms = [z.id for z in evidence if z.tool == "RegionZoom" and z.source == ev.source]
    body = re.sub(r"^(Figure|Fig\.|FIGURE)\s*\w+\s*[:.|]?\s*", "", ev.content)   # drop "Figure 3:"
    what, *rest = sentences(body) or [ev.content]
    claims = [{"text": f"{ev.label} plots the {what[0].lower() + what[1:]}",
               "cites": [{"evidence": ev.id, "quote": what}]}]
    for s in rest:
        claims.append({"text": s, "cites": [{"evidence": i, "quote": s} for i in [ev.id] + zooms],
                       "conclusion": True})
    return claims


def draft_claims(state: AgentState) -> list:
    # Model stand-in for N9: claims from templates, each citing its evidence.
    evidence, entities = state["evidence"], state["query_filters"]["entities"]
    corpus = get_corpus(state["paper_id"])
    claims = []
    for ev in evidence:
        if ev.kind == "text":
            claims += text_claims(ev, corpus)
        elif ev.kind == "table":
            claims += table_claims(ev, entities)
        elif ev.tool == "FigureRetriever":
            claims += figure_claims(ev, evidence)
    return claims


def fabricate(claims: list) -> list:
    # Misquote one number by +1: the first cited table cell if there is one,
    # otherwise the first quoted number. Prose and citation change together, so
    # they agree -- the failure mode G2 exists to catch.
    claims = [dict(c, cites=[dict(x) for x in c["cites"]]) for c in claims]   # never edit originals
    for kind in ("value", "quote"):
        for claim in claims:
            for cite in claim["cites"]:
                m = re.search(r"\d+(?:\.\d+)?", cite.get(kind, ""))
                if not m:
                    continue
                old = m.group()
                new = f"{float(old) + 1:.1f}" if "." in old else str(int(old) + 1)
                cite[kind] = cite[kind].replace(old, new, 1)
                claim["text"] = claim["text"].replace(old, new, 1)
                claim["fabricated"] = True
                return claims
    return claims


def n9_grounded_generator(state: AgentState) -> dict:
    # N9 -- Grounded Generator. Writes the answer from the buffer ONLY: it has no
    # access to the corpus, and every claim cites the evidence it rests on.
    #
    # Real version (BUILD_PLAN.md Day 6): Qwen2.5-VL with the buffer (text + image
    # crops) in context, asked for JSON claims with citations; on a rewrite, the
    # rejected claims are sent back with G2's reasons.
    assert state.get("evidence"), "N9: the buffer is empty -- G1 must never let this through"
    claims = draft_claims(state)
    if cfg.HALLUCINATE_ALWAYS or (cfg.HALLUCINATE_ONCE and state.get("drafts", 0) == 0):
        claims = fabricate(claims)
    n = state.get("drafts", 0) + 1
    lie = "   <-- MISQUOTED" if any(c.get("fabricated") for c in claims) else ""
    return {"claims": claims, "drafts": n,
            "events": step("N9", f"draft {n}: {len(claims)} claim(s){lie}", short=f"draft {n}")}


# --- G2: the Faithfulness Gate ------------------------------------------------------

def cite_context(cite: dict, ev: Evidence) -> str:
    # Everything a citation vouches for: the quote or the cell, and where it sits.
    parts = [ev.label, f"Section {ev.section}" if ev.section else "",
             cite.get("quote", ""), cite.get("value", ""), *cite.get("cell", ())]
    return " ".join(p for p in parts if p)


def check_claim(claim: dict, evidence: dict) -> Optional[str]:
    # Why a claim is unsupported, or None if every part of it checks out.
    if not claim["cites"]:
        return "cites nothing"
    vouched = []
    for cite in claim["cites"]:
        ev = evidence.get(cite["evidence"])
        if ev is None:
            return f"cites {cite['evidence']}, which is not in the buffer"
        if "quote" in cite and cite["quote"] not in ev.content:
            return f"the quote is not in {ev.id}"
        if "cell" in cite:
            row, col = cite["cell"]
            actual = next((c["value"] for c in ev.cells if (c["row"], c["col"]) == (row, col)), None)
            if actual != cite["value"]:
                return f"{ev.id} {row}/{col} is {actual}, the claim says {cite['value']}"
        vouched.append(cite_context(cite, ev))
    known = set(NUMBER.findall(" ".join(vouched)))
    unvouched = [n for n in NUMBER.findall(claim["text"]) if n not in known]
    if unvouched:
        return f"the number {', '.join(unvouched)} is in no citation"
    return None


def g2_faithfulness_check(state: AgentState) -> dict:
    # G2 -- Faithfulness Gate. Checks every claim against the buffer, the only
    # source the generator may use. Writes the failure list, REPLACING it.
    claims = state.get("claims")
    assert claims is not None, "G2: N9 has not run (no claims in state)"
    evidence = {e.id: e for e in state["evidence"]}
    failures = []
    for i, claim in enumerate(claims):
        if claim.get("rejected"):
            continue
        reason = check_claim(claim, evidence)
        if reason:
            failures.append({"claim": i, "text": claim["text"], "reason": reason})
    for f in failures:
        trace("G2", f"REJECT claim {f['claim'] + 1}: {f['reason']}")
    msg = (f"{len(failures)} of {len(claims)} claim(s) unsupported" if failures
           else f"all {len(claims)} claim(s) verified against the buffer")
    return {"faith_failures": failures,
            "events": step("G2", msg, short="G2 reject" if failures else "G2 pass")}


def faithfulness_gate(state: AgentState):
    # G2's routing. Returns route names only -- a LIST when the draft is faithful,
    # because then all three proof nodes run at once (LangGraph's fan-out).
    if not state.get("faith_failures"):
        return ["links", "boxes", "trace"]
    if state.get("rewrite_count", 0) >= cfg.MAX_REWRITES:
        trace("GATE", f"rewrite budget spent ({cfg.MAX_REWRITES}) -> dropping claims")
        return "exhausted"
    return "rewrite"


def rewrite_prep(state: AgentState) -> dict:
    # Body of the regenerate edge: owns the counter that bounds the loop.
    n = state.get("rewrite_count", 0) + 1
    bad = [f["claim"] + 1 for f in state["faith_failures"]]
    return {"rewrite_count": n,
            "events": step("REWRT", f"pass {n}/{cfg.MAX_REWRITES}: rewrite claim(s) {bad}")}


def drop_unsupported(state: AgentState) -> dict:
    # Budget spent: withhold the claims that still fail rather than publish them.
    # Marks rather than deletes, so the report can say plainly what was withheld.
    bad = {f["claim"] for f in state["faith_failures"]}
    claims = [dict(c, rejected=True) if i in bad else c for i, c in enumerate(state["claims"])]
    return {"claims": claims, "status": "unsupported_claims_dropped",
            "events": step("DROP", f"withholding {len(bad)} unsupported claim(s)", short="drop")}


def gate_until_proofs(state: AgentState) -> str:
    # Notebook Section 9 only: the proof nodes do not exist yet, so a faithful
    # draft ends the run.
    route = faithfulness_gate(state)
    return "faithful" if isinstance(route, list) else route
