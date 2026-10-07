# N5: the agentic planner -- the only node that decides what happens next.
from gea import config as cfg
from gea.llm import llm_plan
from gea.state import AgentState
from gea.trace import step, trace


def rule_based_action(state: AgentState) -> dict:
    # The planner's stand-in logic: work on the first sub-goal G1 reported as
    # uncovered, with the tool for its modality; if that tool was already tried for
    # this sub-goal, move on to the next modality.
    goals, verdict = state["sub_goals"], state.get("sufficiency")
    uncovered = verdict["uncovered"] if verdict else list(range(len(goals)))
    target = uncovered[0] if uncovered else 0
    goal = goals[target]
    order = [goal["modality"]] + [m for m in cfg.MODALITIES if m != goal["modality"]]
    tool = next((m for m in order if (target, m) not in state.get("tried", [])), goal["modality"])
    extra = [e for e in state["query_filters"]["entities"] if e.lower() not in goal["text"].lower()]
    return {"tool": tool, "query": " ".join([goal["text"], *extra]), "sub_goal": target,
            "reason": f"sub-goal {target + 1} needs {goal['modality']} evidence"}


def propose_action(state: AgentState) -> dict:
    # Stage 1 of N5: which tool to call next, before any rule is applied.
    #
    # PLANNER_MODE "rules": rule_based_action. PLANNER_MODE "llm": the local model
    # (llm.llm_plan). If the model cannot be reached or replies badly, the rules are
    # used instead and the run records that it happened.
    assert cfg.PLANNER_MODE in ("rules", "llm"), f"unknown PLANNER_MODE {cfg.PLANNER_MODE!r}"
    rules = rule_based_action(state)
    if cfg.PLANNER_MODE == "llm":
        try:
            return {**rules, **llm_plan(state), "planner": "llm"}
        except (OSError, ValueError, KeyError) as e:        # unreachable, or a bad reply
            trace("N5", f"LLM planner failed ({type(e).__name__}) -> rules")
            return {**rules, "planner": "rules (LLM unavailable)"}
    return {**rules, "planner": "rules"}


def n5_agentic_planner(state: AgentState) -> dict:
    # N5 -- Agentic Planner. The only node that decides what happens next: it picks
    # ONE retrieval tool per round, and never adds evidence itself.
    #
    # Two stages, kept apart on purpose: propose_action says which tool, then the
    # rules here decide -- the tool must exist, and a (sub-goal, tool) pair that
    # already ran is not repeated. The rules hold whoever proposed.
    assert state.get("sub_goals"), "N5: N1 has not run (no sub_goals in state)"
    assert state.get("query_filters") is not None, "N5: N2 has not run (no query_filters in state)"
    action = propose_action(state)
    if action["tool"] not in cfg.MODALITIES or (action["sub_goal"], action["tool"]) in state["tried"]:
        action = {**rule_based_action(state), "planner": action["planner"] + ", overruled by rules"}
    missing = (state.get("sufficiency") or {}).get("missing")
    msg = (f"round {state['round']}: {action['tool']} <- \"{action['query']}\""
           + (f"  [G1: missing {missing}]" if missing else ""))
    return {"action": action, "tried": state["tried"] + [(action["sub_goal"], action["tool"])],
            "events": step("N5", msg)}
