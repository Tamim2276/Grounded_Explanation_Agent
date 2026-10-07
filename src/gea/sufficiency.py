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


def g1_sufficiency_check(state: AgentState) -> dict:
    # G1 -- Sufficiency Gate: can the question be answered from the buffer?
    #
    # Real version (BUILD_PLAN.md Day 5): the vision-language model as judge,
    # returning JSON {"sufficient": bool, "missing": str}. The stand-in: every
    # sub-goal must be covered, and every name the question insists on must appear
    # in the evidence. Writes the verdict; the routing is done by sufficiency_gate.
    goals, evidence = state.get("sub_goals"), state.get("evidence", [])
    assert goals, "G1: N1 has not run (no sub_goals in state)"
    corpus = get_corpus(state["paper_id"])
    uncovered = [i for i, g in enumerate(goals) if not covers(evidence, g, corpus)]
    text = " ".join(e.content.lower() for e in evidence)
    absent = [x for x in state["query_filters"]["entities"] if x.lower() not in text]
    missing = ([f"{goals[i]['modality']} evidence for sub-goal {i + 1}" for i in uncovered]
               + [f"any mention of '{x}'" for x in absent])
    verdict = {"sufficient": not missing, "uncovered": uncovered, "missing": "; ".join(missing)}
    msg = "SUFFICIENT" if not missing else f"INSUFFICIENT: missing {verdict['missing']}"
    return {"sufficiency": verdict,
            "events": step("G1", msg, short="G1 pass" if not missing else "G1 reject")}


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
