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
from gea.text import CLAIM_NUMBER, score, sentences
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
    caption = ev.content.splitlines()[0]                  # the caption; later lines are the figure's words/values
    body = re.sub(r"^(Figure|Fig\.|FIGURE)\s*\w+\s*[:.|]?\s*", "", caption)   # drop "Figure 3:"
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


# --- N9: the model writer (cfg.LLM_MODE = "llm", BUILD_PLAN.md Day 6) ---------------

N9_SYSTEM = (                                   # notes/prompts.md
    "You answer a question about a scientific paper using ONLY the evidence items given "
    "(E1, E2, ...); you know nothing else about the paper. Write 1 to 4 short claims. Every claim "
    "cites the items it rests on, in one of three ways:\n"
    "- quote: copy exact words from an item (a paragraph, a caption, or a figure's printed words);\n"
    "- cell: ONE value of a table or chart, by its exact row and column names, with the value "
    "copied (cite several cells for several values);\n"
    "- observation: what you see in an attached figure image, in words (figures only).\n"
    "Never write a number that is not in a quote or cell you cite: do not read exact numbers off "
    "images. The 'Values (cite as cells)' lines are not text of the paper: cite them as cells, "
    "never quote them. Mark the claim that directly answers the question with conclusion = true. If the "
    "evidence does not answer the question, say so in one claim that quotes the closest item.")


def cells_text(ev: Evidence, limit: int = 2500) -> str:
    # A table's or chart's values with their exact row and column names, for citing.
    rows = {}
    for c in ev.cells:
        if c["value"] == c["row"]:                     # the row's own label cell, not a value
            continue
        rows.setdefault(c["row"], []).append(f"'{c['col']}' = {c['value']}")
    text = "\n".join(f"  row '{r}': " + "; ".join(v) for r, v in rows.items())
    return text if len(text) <= limit else text[:limit] + " ..."


def describe_for_writer(ev: Evidence, image: bool) -> str:
    where = ev.label or f"Section {ev.section}"
    head = f"[{ev.id}] ({ev.kind}, {where}, page {ev.page})"
    if ev.tool == "RegionZoom":
        return f"{head}: a zoomed view of {ev.label}" + (" (image attached)" if image else "")
    lines = ev.content.splitlines()
    if ev.cells:          # a table, or a chart read from the PDF: caption, printed words, then the values
        body = "\n".join([lines[0]] + [l for l in lines[1:] if l.startswith("Words printed")]
                         + ["Values (cite as cells):", cells_text(ev)])
    else:
        body = ev.content if len(ev.content) <= 2500 else ev.content[:2500] + " ..."
    return f"{head}{' (image attached)' if image else ''}: {body}"


def writer_schema(ids: list, cell_ids: list = None, figure_ids: list = None) -> dict:
    # The answer's shape. `evidence` must be one of the buffer's ids, so the model cannot
    # cite an item that is not there; a cell cite only items with cells (tables, charts
    # read from the PDF), an observation only figures.
    def ev(allowed):
        return {"type": "string", "enum": allowed}
    options = [{"type": "object", "required": ["evidence", "quote"],
                "properties": {"evidence": ev(ids), "quote": {"type": "string"}}}]
    if cell_ids is None or cell_ids:
        options.append({"type": "object", "required": ["evidence", "row", "col", "value"],
                        "properties": {"evidence": ev(ids if cell_ids is None else cell_ids), "row": {"type": "string"},
                                       "col": {"type": "string"}, "value": {"type": "string"}}})
    if figure_ids is None or figure_ids:
        options.append({"type": "object", "required": ["evidence", "observation"],
                        "properties": {"evidence": ev(ids if figure_ids is None else figure_ids),
                                       "observation": {"type": "string"}}})
    cite = {"anyOf": options}
    claim = {"type": "object", "required": ["text", "cites", "conclusion"],
             "properties": {"text": {"type": "string"}, "conclusion": {"type": "boolean"},
                            "cites": {"type": "array", "minItems": 1, "maxItems": 4, "items": cite}}}
    return {"type": "object", "required": ["claims"],
            "properties": {"claims": {"type": "array", "minItems": 1, "maxItems": 4, "items": claim}}}


def clean_name(name: str) -> str:
    # A row or column name as the model copied it from the values list -- "row '# 4-hop
    # triples'", "'Bing-News'" -- back to the name itself.
    name = name.strip()
    if name.lower().startswith("row "):
        name = name[4:]
    return name.strip().strip("'\"‘’“”").strip()


def as_cite(c: dict) -> dict:
    # The model's cite -> the internal shape G2 and N10 read.
    if "row" in c:
        return {"evidence": c["evidence"], "cell": (clean_name(c["row"]), clean_name(c["col"])),
                "value": c["value"].strip()}
    if "quote" in c:
        return {"evidence": c["evidence"], "quote": c["quote"].strip()}
    return {"evidence": c["evidence"], "observation": c["observation"].strip()}


def llm_draft(state: AgentState) -> list:
    from gea.llm import call_llm
    from gea.sufficiency import evidence_images
    evidence = state["evidence"]
    images = evidence_images(state, limit=2)
    shown = {ev.source for ev, _ in images}
    user = (f"Question: {state['question']}\nEvidence:\n"
            + "\n".join(describe_for_writer(ev, ev.source in shown and ev.tool in ("FigureRetriever", "RegionZoom"))
                        for ev in evidence))
    failures = state.get("faith_failures") or []
    kept = []
    if state.get("rewrite_count", 0) and failures:          # a rewrite: keep what passed, fix what failed
        previous, failed = state.get("claims", []), {f["claim"] for f in failures}
        kept = [c for i, c in enumerate(previous) if i not in failed and not c.get("rejected")]
        if kept:
            user += "\n\nThese claims are already verified and kept:\n" + "\n".join(f"- {c['text']}" for c in kept)
        user += ("\n\nThese claims failed the checks; write corrected claims to replace them:\n"
                 + "\n".join(f"- \"{previous[f['claim']]['text']}\": {f['reason']}" for f in failures))
    schema = writer_schema([e.id for e in evidence], [e.id for e in evidence if e.cells],
                           [e.id for e in evidence if e.kind == "figure"])
    answer = call_llm([{"role": "system", "content": N9_SYSTEM}, {"role": "user", "content": user}],
                      schema, "n9_claims", images=tuple(png for _, png in images))
    return kept + [{"text": c["text"].strip(), "cites": [as_cite(x) for x in c["cites"]],
                    "conclusion": bool(c.get("conclusion"))} for c in answer["claims"]]


def n9_grounded_generator(state: AgentState) -> dict:
    # N9 -- Grounded Generator. Writes the answer from the buffer ONLY: it has no
    # access to the corpus, and every claim cites the evidence it rests on.
    #
    # cfg.LLM_MODE = "llm" (Day 6): Qwen2.5-VL with the buffer (text, table and chart
    # values, figure images) in context, answering in JSON claims with citations; on a
    # rewrite the rejected claims are sent back with G2's reasons. "rules": templates.
    from gea.question import model_or_rules
    assert state.get("evidence"), "N9: the buffer is empty -- G1 must never let this through"
    claims, how = model_or_rules(llm_draft, draft_claims, state)
    if cfg.HALLUCINATE_ALWAYS or (cfg.HALLUCINATE_ONCE and state.get("drafts", 0) == 0):
        claims = fabricate(claims)
    n = state.get("drafts", 0) + 1
    lie = "   <-- MISQUOTED" if any(c.get("fabricated") for c in claims) else ""
    return {"claims": claims, "drafts": n,
            "events": step("N9", f"draft {n}: {len(claims)} claim(s){how}{lie}", short=f"draft {n}")}


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
    known = set(CLAIM_NUMBER.findall(" ".join(vouched)))
    unvouched = [n for n in CLAIM_NUMBER.findall(claim["text"]) if n not in known]
    if unvouched:
        return f"the number {', '.join(unvouched)} is in no citation"
    return None


SUPPORT_SYSTEM = (                              # notes/prompts.md
    "You check claims about a scientific paper against the evidence each one cites. A claim is "
    "supported only if the cited evidence says it AS WRITTEN: the same subject, row, column, "
    "setting and number. A correct value attached to the wrong subject or row is NOT supported, "
    "and neither is a claim that goes beyond its evidence. Judge only from the cited evidence; "
    "figure images are attached where a claim cites a figure. Give a verdict for every claim.")
SUPPORT_SCHEMA = {"type": "object", "required": ["verdicts"],
                  "properties": {"verdicts": {"type": "array", "items": {
                      "type": "object", "required": ["claim", "supported", "reason"],
                      "properties": {"claim": {"type": "integer"}, "supported": {"type": "boolean"},
                                     "reason": {"type": "string"}}}}}}


def show_cite(cite: dict, ev: Evidence) -> str:
    if "cell" in cite:
        return f"{ev.id} ({ev.label}): row '{cite['cell'][0]}', column '{cite['cell'][1]}' = {cite['value']}"
    if "quote" in cite:
        return f"{ev.id} ({ev.label or 'Section ' + str(ev.section)}): quote \"{cite['quote']}\""
    return f"{ev.id} ({ev.label}): observation \"{cite['observation']}\""


def support_check(claims: list, numbers: list, state: AgentState) -> dict:
    # G2's layer 2: one model call over every claim that passed the equality tests.
    # Returns {claim index: reason} for the claims the model finds unsupported (a missing
    # verdict counts as unsupported).
    from gea.llm import call_llm
    from gea.sufficiency import evidence_images
    evidence = {e.id: e for e in state["evidence"]}
    cited = sorted({c["evidence"] for i in numbers for c in claims[i]["cites"]}, key=lambda x: int(x[1:]))
    images = [(ev, png) for ev, png in evidence_images(state, limit=2) if ev.id in cited]
    user = ("Claims:\n" + "\n".join(f"{k}. \"{claims[i]['text']}\" -- cites: "
                                    + "; ".join(show_cite(c, evidence[c['evidence']]) for c in claims[i]["cites"])
                                    for k, i in enumerate(numbers, 1))
            + "\n\nCited evidence:\n" + "\n".join(describe_for_writer(evidence[e], any(ev.id == e for ev, _ in images))
                                                  for e in cited))
    answer = call_llm([{"role": "system", "content": SUPPORT_SYSTEM}, {"role": "user", "content": user}],
                      SUPPORT_SCHEMA, "g2_support", images=tuple(png for _, png in images))
    verdicts = {v["claim"]: v for v in answer["verdicts"]}
    return {i: ("the checker gave no verdict" if k not in verdicts else verdicts[k]["reason"])
            for k, i in enumerate(numbers, 1) if k not in verdicts or not verdicts[k]["supported"]}


def support_check_on() -> bool:
    return cfg.SUPPORT_CHECK == "on" or (cfg.SUPPORT_CHECK == "auto" and cfg.LLM_MODE == "llm")


def g2_faithfulness_check(state: AgentState) -> dict:
    # G2 -- Faithfulness Gate. Checks every claim against the buffer, the only
    # source the generator may use. Writes the failure list, REPLACING it.
    #
    # Layer 1, equality tests: the cited item exists, a quote is word for word, a cell's
    # value matches, and every number in the claim is in a citation. Layer 2 (Day 6), the
    # support check: the model judges whether the cited evidence supports the claim AS
    # WRITTEN -- which catches a real cell from the wrong row, and judges figure
    # observations. A claim fails if either layer fails.
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
    passed = [i for i, c in enumerate(claims) if not c.get("rejected") and i not in {f["claim"] for f in failures}]
    if passed and support_check_on():
        try:
            for i, why in support_check(claims, passed, state).items():
                failures.append({"claim": i, "text": claims[i]["text"], "reason": f"support check: {why}"})
        except (OSError, ValueError, KeyError) as e:             # no model: layer 1 alone, said so
            trace("G2", f"support check unavailable ({type(e).__name__}) -> equality tests only")
        failures.sort(key=lambda f: f["claim"])
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
