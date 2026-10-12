# G1, the Sufficiency Gate, and the bounded retrieval loop around it.
#
# The red diamond of thesis Figure 4.1 is two pieces in code: a node that WRITES a
# verdict (g1_sufficiency_check) and a conditional function that READS it and
# returns a route (sufficiency_gate). A conditional function cannot write state, so
# the loop's counter lives in its own node, replan_prep.
from gea import config as cfg
from gea.corpus import get_corpus
from gea.state import AgentState
from gea.text import score
from gea.trace import step, trace


def covers(evidence: list, goal: dict, corpus: dict) -> bool:
    # A sub-goal is covered by evidence of the right kind that shares words with it.
    # A text sub-goal can be answered by any kind; a table or figure one needs its own.
    kinds = {"text": {"text", "table", "figure"}, "table": {"table"}, "figure": {"figure"}}
    return any(e.kind in kinds[goal["modality"]] and score(goal["text"], e.content, corpus) > 0
               for e in evidence)


def rule_verdict(state: AgentState) -> dict:
    # The stand-in judge: every sub-goal must be covered, and every name the question
    # insists on must appear in the evidence.
    goals, evidence = state["sub_goals"], state.get("evidence", [])
    corpus = get_corpus(state["paper_id"])
    uncovered = [i for i, g in enumerate(goals) if not covers(evidence, g, corpus)]
    text = " ".join(e.content.lower() for e in evidence)
    absent = [x for x in state["query_filters"]["entities"] if x.lower() not in text]
    missing = ([f"{goals[i]['modality']} evidence for sub-goal {i + 1}" for i in uncovered]
               + [f"any mention of '{x}'" for x in absent])
    return {"sufficient": not missing, "uncovered": uncovered, "missing": "; ".join(missing)}


G1_SYSTEM = (                                   # v7 (notes/prompts.md; v8 was tried and dropped)
    "You check whether the evidence collected so far is enough to answer a question about a "
    "scientific paper. Say sufficient only if, for EVERY sub-question, some evidence item contains "
    "the specific fact, number or picture that answers it -- not merely the same topic. The fact "
    "must BE the answer: for 'which X' it names the X; for 'how does A affect B' or 'what is the "
    "effect of A' it states the direction or size of the change; for 'how many' or 'what value' "
    "it gives the number. A fact that only restates the question or describes the topic does not "
    "answer it. First give, for each sub-question you can answer, the id of the item (E1, E2, ...) "
    "and the answering fact in a few words; then list the numbers of the sub-questions still "
    "uncovered, say in a few words what is missing, and only then decide.")
# Written in this order -- the facts first, the verdict last -- because llama.cpp makes the model
# write the fields in schema order: a verdict written first is only justified afterwards (dev, Day 7).
# maxLength: the 7B once repeated itself inside "missing" for 5,800 tokens (88 s) until the
# reply broke off as invalid JSON; llama.cpp enforces the limit token by token.
G1_SCHEMA = {"type": "object", "required": ["answers", "uncovered", "missing", "sufficient"],
             "properties": {"answers": {"type": "array", "items": {
                                "type": "object", "required": ["sub_question", "evidence", "fact"],
                                "properties": {"sub_question": {"type": "integer"}, "evidence": {"type": "string"},
                                               "fact": {"type": "string", "maxLength": 300}}}},
                            "uncovered": {"type": "array", "items": {"type": "integer"}},
                            "missing": {"type": "string", "maxLength": 300},
                            "sufficient": {"type": "boolean"}}}


def describe_evidence(ev, limit: int = 1500) -> str:
    # One buffer item as the judge reads it.
    where = f"{ev.label or 'Section ' + str(ev.section)}, page {ev.page}"
    body = ev.content if len(ev.content) <= limit else ev.content[:limit] + " ..."
    image = " (image attached)" if ev.kind == "figure" and ev.tool == "FigureRetriever" else ""
    return f"{ev.id} [{ev.kind}, {where}]{image}: {body}"


def evidence_images(state: AgentState, limit: int = 2) -> list:
    # (figure evidence, PNG) for the figures in the buffer: the zoomed crop if N7 made one,
    # otherwise the region re-rendered. Real backend only (the stub has no PDF).
    if cfg.BACKEND != "real":
        return []
    from gea.retrieval import zoom_crop
    corpus, images = get_corpus(state["paper_id"]), []
    for ev in [e for e in state["evidence"] if e.tool == "FigureRetriever"][:limit]:
        zoom = next((z.crop for z in state["evidence"] if z.tool == "RegionZoom" and z.source == ev.source), None)
        images.append((ev, zoom if isinstance(zoom, bytes) else zoom_crop(corpus, ev)[0]))
    return images


def figure_images(state: AgentState, limit: int = 2) -> list:
    # The judge sees figure images only when a sub-goal needs a figure.
    if not any(g["modality"] == "figure" for g in state["sub_goals"]):
        return []
    return [png for _, png in evidence_images(state, limit)]


def llm_verdict(state: AgentState) -> dict:
    # The model as judge, then the RULE FLOOR: "sufficient" counts only if every table or
    # figure sub-goal has at least one item of that kind in the buffer. A lenient judge
    # cannot wave through an answer whose table was never retrieved.
    from gea.llm import call_llm
    goals, evidence = state["sub_goals"], state.get("evidence", [])
    user = (f"Question: {state['question']}\nSub-questions:\n"
            + "\n".join(f"{i}. [{g['modality']}] {g['text']}" for i, g in enumerate(goals, 1))
            + "\nEvidence:\n" + "\n".join(describe_evidence(e) for e in evidence))
    answer = call_llm([{"role": "system", "content": G1_SYSTEM}, {"role": "user", "content": user}],
                      G1_SCHEMA, "g1_sufficiency", images=tuple(figure_images(state)))
    kinds = {e.kind for e in evidence}
    floor = [i for i, g in enumerate(goals) if g["modality"] in ("table", "figure") and g["modality"] not in kinds]
    unbacked = [i for i in range(len(goals)) if answer["sufficient"] and not backed(answer, i + 1, evidence)]
    if answer["sufficient"] and not floor and not unbacked:
        return {"sufficient": True, "uncovered": [], "missing": "",
                "answers": [{k: a[k] for k in ("sub_question", "evidence", "fact")} for a in answer["answers"]]}
    uncovered = sorted({i - 1 for i in answer["uncovered"] if 1 <= i <= len(goals)} | set(floor) | set(unbacked)) or [0]
    missing = "; ".join([f"{goals[i]['modality']} evidence for sub-goal {i + 1} (rule floor)" for i in floor]
                        + [f"the judge named no item that backs sub-goal {i + 1}" for i in unbacked]
                        + ([answer["missing"]] if not answer["sufficient"] and answer["missing"] else []))
    return {"sufficient": False, "uncovered": uncovered, "missing": missing or "the model found a gap"}


def backed(answer: dict, n: int, evidence: list) -> bool:
    # The judge's "sufficient" holds for sub-question n only if it named an item that is in
    # the buffer and a fact whose numbers (if any) appear in that item -- except a figure,
    # whose numbers are in the picture. A vague or invented "yes" does not pass.
    from gea.text import CLAIM_NUMBER, loose
    items = {e.id: e for e in evidence}
    for a in answer["answers"]:
        ev = items.get(a["evidence"].strip())
        if a["sub_question"] == n and ev and a["fact"].strip():
            if ev.kind == "figure" or all(x in loose(ev.content) for x in CLAIM_NUMBER.findall(a["fact"])):
                return True
    return False


def g1_sufficiency_check(state: AgentState) -> dict:
    # G1 -- Sufficiency Gate: can the question be answered from the buffer?
    #
    # cfg.LLM_MODE = "llm" (BUILD_PLAN.md Day 5): the vision-language model judges,
    # with the rule floor above; "rules": the stand-in. Writes the verdict; the routing
    # is done by sufficiency_gate.
    from gea.question import model_or_rules
    goals = state.get("sub_goals")
    assert goals, "G1: N1 has not run (no sub_goals in state)"
    verdict, how = model_or_rules(llm_verdict, rule_verdict, state)
    msg = ("SUFFICIENT" if verdict["sufficient"] else f"INSUFFICIENT: missing {verdict['missing']}") + how
    return {"sufficiency": verdict,
            "events": step("G1", msg, short="G1 pass" if verdict["sufficient"] else "G1 reject")}


def sufficiency_gate(state: AgentState) -> str:
    # G1's routing -- the red diamond. Returns a ROUTE NAME only; it must not return
    # a state update, since LangGraph ignores anything but the route.
    if state["sufficiency"]["sufficient"]:
        return "sufficient"
    if state["round"] >= cfg.MAX_ROUNDS:
        trace("GATE", f"round {state['round']}/{cfg.MAX_ROUNDS} spent -> exhausted")
        return "exhausted"
    return "retrieve_again"


def replan_prep(state: AgentState) -> dict:
    # The body of the red dashed edge back to N5. Owns the round counter -- the part
    # that makes the retrieval loop terminate.
    n = state["round"] + 1
    return {"round": n, "events": step("REPLN", f"round {n}/{cfg.MAX_ROUNDS}: back to the planner")}


def abstain(state: AgentState) -> dict:
    # Graceful exit once the round budget is spent: report the gap, never guess.
    # Keeps the evidence it did find, so the report can say what was found and
    # exactly what is still missing.
    missing = state["sufficiency"]["missing"]
    return {"status": "insufficient_evidence",
            "events": step("ABSTN", f"no answer after {state['round']} round(s); missing {missing}",
                           short="abstain")}
